# Agentic RL 话题笔记：Off-policyness、Sample Efficiency 与 Privileged Information

- 日期：2026-03-02
- 来源：微信公众号「青稞AI」
- 原文链接：https://mp.weixin.qq.com/s/KZUQpS_Hger26thBsLmI-g
- 标题：聊聊 Agentic RL 热门话题：Off-policyness，Sample Efficiency与Priviledge Information

---

## 一、核心问题

Agentic RL 训练里，常见目标彼此冲突：

1. **提高 sample efficiency**（少 rollouts 也能学到）
2. **控制 off-policyness**（训练分布不要偏离当前策略太远）
3. **保证可泛化**（不只在 in-domain 任务上有效）

文章观点：

- 纯 SFT 最“省样本”，但 off-policy 最重，容易导致 coverage 不足和性能崩塌。
- 纯 on-policy 最稳，但 sample efficiency 太低。
- 现实里需要“融合方法”在二者间折中。

---

## 二、从 OPD 出发理解融合方法

### 1) OPD（On-Policy Distillation）

- 用 base policy 的 on-policy rollout，
- 但学习信号来自 teacher policy 的 token-level distillation（如 reverse KL）。

特点：

- 兼顾一定稳定性与效率；
- 监督粒度细；
- 可与 reward-based RL 结合。

### 2) OPSD

- 用 **optimal solution/trajectory** 替代 teacher，
- 可理解为“对 SFT 的 off-policyness 控制版”。

### 3) SDPO

- 不要求 optimal trajectory，
- 使用 **环境反馈（feedback/reward）** 提供更细粒度监督。

---

## 三、Privileged Information 框架

文章把可用监督信号统一看作 privileged information，典型包括：

- 更强的 teacher policy
- trajectory-level reward
- step-level reward
- optimal trajectory / reference solution

可分为两类：

- **Prior 型信号**（更依赖先验监督，如 teacher / optimal solution）
- **Posterior 型信号**（更依赖环境反馈，如 reward）

这个视角有助于统一比较不同 Agentic RL 算法。

---

## 四、用于引导 rollout 的方法

除直接 SFT / distill 之外，privileged info 还能用于 **guide rollout**：

### 1) POPE

- 用 optimal trajectory 作为 rollout prefix，降低后续求解难度。
- 但 prefix 不应直接计入 base policy loss（避免加重 off-policyness）。

### 2) InT（Intervention Training）

- 给定部分正确轨迹 + reference solution；
- 让策略定位**第一个错误步骤**并只修复该步；
- 后续继续 on-policy 生成。

直觉优势：

- intervention 很短、很轻量；
- 提供关键信号同时尽量维持 on-policy；
- 在 sample efficiency 和最终性能上更有潜力。

---

## 五、我的收获（可行动）

1. 设计 Agentic RL pipeline 时，不要二选一（SFT 或 RL），而是把 off-policy 程度当作可调超参。  
2. 训练数据应按“监督强度 × 分布偏移风险”打标签分层混合。  
3. 对困难任务可尝试“短 intervention + 继续 on-policy rollout”的混合训练范式。  
4. 评估不仅看最终准确率，还要看熵变化、coverage 和 zero-advantage ratio 之类的过程指标。

---

## 参考链接（文中提到）

- DigiRL: https://arxiv.org/abs/2406.11896
- Digi-Q: https://arxiv.org/abs/2502.15760
- OPD: https://thinkingmachines.ai/blog/on-policy-distillation/
- OPSD: https://siyan-zhao.github.io/blog/2026/opsd/
- SDPO: https://arxiv.org/abs/2501.01821
- POPE: https://arxiv.org/abs/2601.18779
- InT: https://arxiv.org/abs/2601.14209
