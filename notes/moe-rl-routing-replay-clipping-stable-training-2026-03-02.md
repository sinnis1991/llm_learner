# MoE 模型 RL 稳定训练笔记：Routing Replay + 裁剪

- 日期：2026-03-02
- 来源：微信公众号「GPU集群基础设施」
- 原文链接：https://mp.weixin.qq.com/s/_fxhqXR7t-uBVWPfvzDuJw
- 原文标题：MoE模型RL训练易崩溃？Qwen团队用“Routing Replay + 裁剪”搞定，30B模型实验验证稳定收敛。
- 相关论文：https://arxiv.org/abs/2512.01374

---

## 一、问题背景

LLM 的 RL 训练存在经典错位：

- 奖励通常是 **sequence-level**（整条回答打分）
- 优化通常是 **token-level**（每个 token 算梯度）

这会导致训练稳定性问题，尤其在 MoE 上更明显：

1. **训练-推理不一致**：训练引擎与推理引擎内核差异（如 Megatron vs vLLM）。
2. **策略陈旧性（staleness）**：同一批采样被拆成多个 mini-batch 多次更新，更新时策略已偏离采样策略。
3. **MoE 动态路由放大问题**：路由变化会放大上面两类不一致，导致更容易崩溃。

---

## 二、核心结论（理论层）

论文给出的关键视角：

- token-level 目标可看作 sequence-level 目标的 **一阶近似**；
- 该近似成立依赖两个条件：
  - 训练-推理一致性尽量高；
  - 策略陈旧性尽量低。

据此可解释为什么常见稳定化技巧有效：

- 重要性采样（IS）矫正
- 裁剪（clipping）
- Routing Replay

---

## 三、方法要点

### 1) MiniRL（极简基线）

作者给了一个最小可行基线，在 token-level 目标上做最少改动，强调：

- 保持一阶近似有效性
- 用 IS 矫正 + 稳定化策略提升可训练性

### 2) Routing Replay（针对 MoE）

作用：缓解 MoE 中路由变化导致的训练-推理偏移。  
在 off-policy 程度增加时，与裁剪配合尤为关键。

### 3) 裁剪（clipping）

作用：限制权重或更新幅度，减少训练不稳定。  
结论上看，在 off-policy 场景中，**与 Routing Replay 一起使用是必要条件**。

---

## 四、实验观察（30B MoE，大规模算力）

### On-policy

- MiniRL（带 IS 矫正）表现最好，训练稳定。
- 去掉训练-推理 IS 矫正会快速崩溃（熵急降）。
- 某些长度归一化策略会破坏一阶近似，性能次优。

### Off-policy

- 全局 batch 为 mini-batch 的 2x/4x/8x 时：
  - **Routing Replay + 裁剪都不可少**，少一个都容易崩。
- 不同 replay 变体在不同 off-policy 强度下优劣不同：
  - off-policy 小时一种更优；
  - off-policy 大时另一种更优。

### 冷启动影响

- 不同冷启动模型在稳定 RL 后最终性能趋同。  
- 启示：当训练稳定后，RL 过程本身对最终上限影响更大。

---

## 五、对我有用的实践启发

1. 做 MoE RL 时，优先把“稳定性工程”作为一等公民，而不是最后补救。  
2. 训练配置要显式管理：
   - 训练-推理一致性（引擎/核/路由行为）
   - off-policy 程度（global batch 与 mini-batch 比例）
3. 当 off-policy 不可避免时，默认开启：
   - IS 矫正
   - 裁剪
   - Routing Replay
4. 监控指标除了 reward，还要盯：
   - 熵（entropy）
   - 训练-推理 KL
   - 崩溃前兆（曲线振荡、熵塌陷）

---

## 六、后续可复现动作（To-Do）

1. 在 toy MoE 环境复现“去掉矫正会崩”的最小对照实验。  
2. 设计 2x/4x/8x off-policy 程度对比，并记录各策略组合稳定区间。  
3. 梳理 Replay 变体（文中 R2/R3）的偏差-方差取舍，形成可执行选择准则。
