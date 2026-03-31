# Qwen Pilot 最近 RL 研究发现整理（FIPO / Future-KL）

- 日期：2026-03-31
- 来源：小红书笔记《最近死磕 RL，聊聊我们的发现》
- 原链接：http://xhslink.com/o/4abry54k0V5
- 作者：Henry（Qwen Pilot）

---

## 一、核心结论（一句话版）

**在大模型推理强化学习里，性能上限更依赖“细粒度信用分配（token-level credit assignment）”，而不是对整段 CoT 粗粒度平均奖励。**

---

## 二、笔记中的 3 个关键发现

### 1) RL 其实很“懒”：策略演化极稀疏

笔记称：在超过 98% 的生成步骤里，模型几乎没有变化。

含义：
- RL 并不会“全面重写”基座模型；
- 更像是在少数关键决策点（逻辑分叉口）进行策略修正。

提及论文：
- https://arxiv.org/abs/2603.22446

---

### 2) 方向比幅度重要：别只盯 KL

笔记强调应追踪 **log-prob 差异方向** 来定位优化“导航方向”。

含义：
- 仅看 KL 散度（幅度约束）不足以解释有效改进；
- 对“往哪儿推”的方向性判断更关键。

提及论文：
- https://arxiv.org/abs/2603.22117

---

### 3) “Oops”时刻多于“Aha”时刻（约 3 倍）

现象：
- 长链推理中，模型常出现“已经走到正确答案，后续冗余反思又把自己推翻”的情况；
- 该破坏性 Oops 时刻比自我纠正 Aha 时刻更多。

归因（笔记观点）：
- 标准 RL（如 GRPO）中的粗粒度 credit assignment，
- 奖励被平均到整条推理链，无法精确识别“哪个 token 真在帮忙、哪个 token 在带偏”。

---

## 三、他们提出的方法：FIPO + Future-KL

### 方法动机

用更细粒度信号，衡量每个 token 对后续轨迹的因果贡献，而不是整轨迹统一打分。

### 方法直觉

- **Future-KL**：衡量“当前 token 对后续行为分布的影响”；
- 将该影响用于优势重加权（adv reweight），让有效 token 获得更强学习信号；
- 抑制会导致后续错误反思/路径偏移的 token。

### 作者在评论区的口径（要点）

- 他们用每个 gradient step 的 log p diff 变化判断更新是否有利；
- Future-KL 视作当前 token 对后续 token 行为好/坏影响的加权汇总；
- 进而对 token 对应优势做重加权。

---

## 四、文中给出的结果（待以论文/代码复核）

- 模型：Qwen2.5-32B（纯 RL）
- AIME24：58.0%
- 有效推理长度：可达 10,000+ tokens
- 对外结论：细粒度信用分配是推理能力扩展关键。

---

## 五、相关论文与仓库

- FIPO 论文：https://arxiv.org/abs/2603.19835
- GitHub：https://github.com/qwenpilot/FIPO
- 另外两篇（笔记引用）：
  - https://arxiv.org/abs/2603.22446
  - https://arxiv.org/abs/2603.22117

---

## 六、学习者视角：可直接跟进的阅读顺序

1. 先读 FIPO 摘要+方法图，确认“Future-KL -> advantage reweight”主链路；
2. 对照 GRPO/PPO 原始目标，标出 credit assignment 差异点；
3. 看 ablation：是否确实降低 Oops、提升长链稳定性；
4. 看代码里 reward / sampling / truncation / length 控制细节，避免只学概念不学实现。

---

## 七、备注

本笔记为对公开帖内容的结构化整理，数值与结论建议以论文和代码复现实验为准。
