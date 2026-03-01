# Agentic 训练近期四篇代表工作总结（Tongyi / GLM-5 / ABE / GEM）

- 日期：2026-03-02
- 来源：微信公众号《总结Agentic训练的最近几篇工作》
- 链接：https://mp.weixin.qq.com/s/IKv0xtIgi5EpuroWslLpjw

---

## TL;DR

1. Agentic 训练的两大关键：**高质量合成数据** + **稳定、可扩展的训练环境**。
2. 对 long-horizon agent 任务，**mid-training 非常关键**：普通任务分布无法覆盖真实工具使用轨迹。
3. 环境不稳定和交互成本高，可通过**仿真工具环境（本地 Python/可控工具集）**显著缓解。
4. 仅有初始轨迹不够，数据通常需要 **refine（提升复杂度/歧义/多样性）** 才能真正提升效果。

---

## 1) Tongyi DeepResearch（2025）

核心贡献：

- 强调 **mid-training** 对 agentic 偏好的塑造作用，再用 multi-turn RL 释放能力。
- 构建自动化、可扩展的数据合成 pipeline（覆盖 mid-training 与 post-training）。
- 给出跨阶段环境分层：
  - **Prior World Environment**：稳定、低成本、可扩展，但缺真实反馈。
  - **Simulated Environment**：可控可复现，成本低、迭代快，但有 sim-to-real gap。
  - **Real-world Environment**：反馈最真实，但成本高、风险高。

启发：

- 训练期就要为“工具交互决策”建偏好，而不是只靠后期微调。
- 环境分层是工程上平衡“真实性/稳定性/成本”的可行路线。

---

## 2) GLM-5（2026）

核心贡献（Agentic相关）：

- 再次验证 **mid-training + long-context/agent data 混入**的重要性。
- SFT 阶段支持多种推理/思维组织形态：
  - **Interleaved Thinking**：工具调用前后都显式思考。
  - **Preserved Thinking**：多轮保留并复用既有思考块，适配长任务。
  - **Turn-level Thinking**：按轮次控制是否启用深度推理，平衡成本与效果。
- Agentic RL 上做了多项稳定性优化（如异步框架、采样/分词稳定性等）。

启发：

- 推理“模式控制”本身就是 Agent 工程能力的一部分。
- 训练和推理策略要一起设计，不能割裂。

---

## 3) ABE（Automated Build Environments, 2025）

核心贡献：

- 聚焦“训练环境自动化构建”，提升 tool-use 训练的可扩展性、稳定性与可验证性。
- pipeline 关键环节：
  - 场景拆解（single-hop / multi-hop / 并行变体）
  - 文档与函数映射（子问题 ↔ 工具接口）
  - 工具整合去冗余
  - 复杂度扩展（功能泛化、参数扩展、类型泛化、工具集扩展）
  - 本地部署并注入可验证行为
- 引入可验证奖励机制，综合评估结构完整性与准确性。

启发：

- Agentic RL 的上限，常被环境质量和可验证性卡住。
- ABE 把“环境工程”变成了可规模化的训练资产。

---

## 4) GEM（2026）

核心贡献：

- 从文本自动合成 tool-use 轨迹，降低数据构造门槛。
- 四阶段链路：
  1. 文本过滤（筛掉缺少 multi-step 操作样本）
  2. workflow & tool 抽取（建步骤与工具 schema）
  3. 轨迹生成（系统提示/用户任务/助手/工具响应）
  4. refine（加复杂度、歧义、真实感）

启发：

- 自动化数据工厂可持续产出 agent 训练样本。
- refine 阶段是从“能用”到“好用”的关键。

---

## 横向对比（我的理解）

- **Tongyi / GLM-5**：强调训练范式（尤其 mid-training）的必要性。
- **ABE**：解决“环境”这一底层瓶颈。
- **GEM**：解决“数据”这一供给瓶颈。

一句话总结：

> Agentic 能力提升不只靠模型本体，更靠“数据工厂 + 环境工厂 + 中期训练策略”三者协同。

---

## 后续可执行动作（给自己）

1. 在本仓做一个最小 ABE 风格 toy 环境（2~3 个可验证工具）。
2. 参考 GEM pipeline，为一个垂直任务合成 100 条工具轨迹并做一轮 refine。
3. 设计一个“turn-level thinking 开/关”对比实验，量化成本与成功率变化。
4. 补充原文中 4 篇论文的速读卡（问题、方法、实验、局限、可复现点）。

---

## 参考

- Tongyi DeepResearch Technical Report (arXiv:2510.24701)
- GLM-5: from Vibe Coding to Agentic Engineering (arXiv:2602.15763)
- Unlocking Implicit Experience: Synthesizing Tool-Use Trajectories from Text (arXiv:2601.10355)
- Feedback-Driven Tool-Use Improvements via Automated Build Environments (arXiv:2508.08791)
