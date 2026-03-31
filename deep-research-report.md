# 大模型中 OPD（On-Policy Distillation）的概念、算法与应用研究报告

## 执行摘要

OPD（On-Policy Distillation，中文常译“在策略蒸馏/在线策略蒸馏”）是一类面向自回归大模型（尤其是大语言模型，LLM）的后训练范式：**数据（轨迹/输出序列）由学生模型按自身当前策略生成（on-policy），随后由更强的教师模型在这些“学生会真实去到的前缀/状态”上提供密集监督（通常是 token 级 logits/概率分布），学生据此更新**。其核心动机是缓解传统离线蒸馏/监督微调在自回归生成中固有的“训练—推理分布失配（exposure bias）”，即训练时看到的前缀与推理时自己生成的前缀不同，错误会级联放大。OPD以“学生自生成错误→教师在错误处给纠偏信号”为机制，直接把监督对齐到推理分布，从而在若干总结、翻译、算术推理、长链推理与 agentic 场景中被反复报告为更稳健或更高效的能力迁移方式之一。citeturn18view1turn31view0turn15view0

从算法形态看，OPD可以被理解为：  
- **模仿学习/策略蒸馏的“数据聚合（DAgger-like）”版本**：不断用学生滚动采样，再向教师查询“在这些状态下应如何分配概率”。citeturn18view1turn2search0  
- **与 KL 约束强化学习（KL-constrained RL）的紧密联系**：多篇工作把 teacher–student 的对数似然比视作 token 级“隐式奖励”，从而把 OPD 解释为一种密集奖励的策略优化；并进一步提出“reward scaling / extrapolation”等泛化形式，在某些多教师或强到弱蒸馏设置中据称可使学生逼近甚至超过教师的某些指标上界。citeturn19view3turn28view0

实践上，OPD面临的主要挑战集中在：**教师评估成本（全词表 logits 的算力与显存）、长序列稳定性（ teacher 在学生偏离的前缀上可能不可靠）、token 级目标的偏差—方差权衡、以及在线数据收集的安全与隐私风险**。因此近一年出现了大量“稳定性/效率”改进：从 sampled-token 近似的脆弱性分析与 top‑K 局部支持集匹配（truncated reverse‑KL）、到 Veto 的 logit 空间桥接目标、REOPOLD 的奖励裁剪与熵引导动态采样、以及用“语言化离散评分”替代 logits 以降低显存瓶颈的 OVD 等。citeturn39view0turn31view0turn28view0turn33view0

本报告给出 OPD 的概念定位、数学表述与算法流程（含 Mermaid 流程图）、代表性变体与论文年份、实验基准与指标、工程部署要点、以及开放研究方向。

**本节关键参考（2–4）**  
- On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes（ICLR 2024，提出/系统化 GKD/On-policy distillation 框架与 stop‑grad 实现）citeturn18view1turn18view2  
- Stable On-Policy Distillation through Adaptive Target Reformulation（Veto，2026）citeturn31view0  
- Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes（2026）citeturn15view0turn39view0  
- Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation（G‑OPD/ExOPD，2026）citeturn19view3turn36view2  

## 定义与直观解释

### OPD 的定义与目标

在 LLM 场景中，将自回归生成视为序列决策过程：每个 token 是动作、前缀是状态。OPD的典型定义是：  
1) 对输入/提示 \(x\)（来自数据分布）由学生模型 \(\pi_\theta\) 采样序列 \(y\sim \pi_\theta(\cdot|x)\)；  
2) 在学生生成的每个前缀 \(h_t=(x,y_{<t})\) 上，让教师模型 \(\pi^\*\) 给出下一 token 的分布（logits/softmax）；  
3) 以某种分布差异度量 \(D(\pi^\*\|\pi_\theta)\)（常见为 KL、reverse KL、JSD 或其截断/加权形式）构造 token 级损失，更新学生。citeturn18view2turn16view1turn31view0

目标可概括为三点：  
- **缓解训练—推理分布失配**：学生训练时看到的“上下文/前缀”来自自身生成而非教师轨迹，因此更接近部署时分布。citeturn18view0turn34view0turn31view0  
- **把“纠错信号”对齐到学生真实错误**：教师在学生产生错误 token 的前缀上给出“本应如何分配概率”的信息，使学习信号更直接。citeturn18view2turn31view0  
- **提供密集、可控的学习信号**：相较只在序列末端给奖励的 RL，OPD常被描述为“密集监督/密集奖励”的替代或补充。citeturn19view3turn16view0  

### OPD 在强化学习与策略蒸馏谱系中的位置

- **策略蒸馏/模仿学习视角**：OPD可视作“on-policy imitation / DAgger 风格的数据聚合”：不断用学生访问状态，再由专家（教师）提供标注并训练。GKD 在动机上就明确借鉴了 imitation learning 与 DAgger 的思路：用学生生成序列降低级联误差与失配。citeturn18view1turn2search0  
- **强化学习视角**：多篇近期工作把 teacher–student 的对数似然比 \(\log \pi^\*(y_t|h_t)-\log \pi_\theta(y_t|h_t)\) 解释为 token 级“隐式奖励/优势”，从而把 OPD 纳入 KL 约束策略优化框架，并将 stop‑gradient 看作降低方差的控制变量。citeturn19view3turn28view0turn31view0  

### 与离线蒸馏、行为克隆、知识蒸馏、在线蒸馏的区别

- **离线蒸馏（off-policy distillation）**：训练序列 \(y\) 多来自教师采样或固定数据集（ground truth/teacher traces）。这类方法在自回归生成中容易出现 exposure bias：训练时依赖“正确前缀”，部署时却在“自生成前缀”上滚动，误差会放大。citeturn18view0turn31view0  
- **行为克隆（Behavior Cloning, BC）**：通常指用专家动作的硬标签做监督学习（交叉熵），在 LLM 上对应 SFT（监督微调）。它不一定需要教师 logits，因此成本更低，但同样可能有分布失配与长程级联误差。citeturn31view0turn18view0  
- **知识蒸馏（KD）**：一般指用教师的软分布（logits）作为更丰富的监督信号。OPD是 KD 的一种“采样分布改造”：监督仍然是教师分布，但采样轨迹来自学生。citeturn18view0turn31view0  
- **在线蒸馏（online distillation）**：术语在文献中不够统一。工程上常指“数据流式到来/模型持续更新（continual/online learning）”，而 OPD 的 “on-policy”强调的是**采样分布来自当前学生策略**；两者可叠加（例如在持续学习中用 OPD 把在线经验巩固进参数），但概念维度不同。OPCD 与相关“在线经验学习”工作明确把 OPD用作在线循环的一环。citeturn34view0turn23search15  

**本节关键参考（2–4）**  
- DAgger / Dataset Aggregation（将模仿学习归约为 no-regret 在线学习）citeturn2search0  
- GKD / On-policy distillation（ICLR 2024）citeturn18view1turn18view2  
- Veto（讨论 on-policy KD 中 forward KL 梯度爆炸与 reverse KL 模式坍塌）citeturn31view0  
- G‑OPD/ExOPD（把 OPD 解释为 KL 约束 RL 的特例并引入 reward scaling）citeturn19view3turn19view2  

## 数学表述与算法流程

### 关键符号与基本目标函数

令：  
- \(x \sim \mathcal{D}\)：提示/输入分布；  
- 学生策略（LLM）\(\pi_\theta(y|x)=\prod_{t=1}^T \pi_\theta(y_t|x,y_{<t})\)；  
- 教师策略 \(\pi^\*(y|x)\)（通常更大或更强）；  
- 前缀/状态 \(h_t=(x,y_{<t})\)；  
- 发散度 \(D(\cdot,\cdot)\)：常用 KL、reverse KL、JSD 或 f-divergence；  
- 训练中常用 stop‑gradient：对“采样分布”不回传梯度，只对学生 logits 回传。citeturn18view2turn16view1  

**离线（supervised/off-policy）KD 的典型形式**：  
\[
\min_\theta \ \mathbb{E}_{(x,y)\sim(\mathcal{X},\mathcal{Y})}\Big[\sum_t D\big(\pi^\*(\cdot|x,y_{<t}),\pi_\theta(\cdot|x,y_{<t})\big)\Big]
\]
其中 \(y\) 来自固定数据集（真值）或教师采样。citeturn18view0turn31view0  

**OPD 的核心形式**（学生 on-policy 采样）：  
\[
\min_\theta \ \mathbb{E}_{x\sim \mathcal{X},\ y\sim \pi_\theta(\cdot|x)}\Big[\sum_t D\big(\pi^\*(\cdot|h_t),\pi_\theta(\cdot|h_t)\big)\Big]
\]
并且在实践实现中常采用“不对 \(y\sim \pi_\theta\) 的采样过程回传梯度”的近似，以获得稳定与高效。GKD明确指出不回传采样梯度与 on-policy imitation 类似，能提升稳定性与计算效率。citeturn18view1turn18view2  

### stop‑gradient 下的梯度估计与“序列级—token级”权衡

**stop‑gradient（类 DAgger）更新**可视为：把当前学生采样得到的序列当作“训练数据”，对每个前缀做教师分布监督并反传到学生 logits；不需要 REINFORCE 项，因此方差较小。citeturn18view2turn16view1  

但当把 OPD理解为“最小化序列级 reverse‑KL”或“最大化对数似然比”时，会出现**token 级近似的偏差—方差权衡**：  
- 近期对“sampled-token OPD”做了系统剖析：token 级 OPD 相对序列级 reverse‑KL 有偏，但长序列下其最坏情况方差上界增长更慢；而更强的未来回报耦合会带来更大梯度方差与不稳定。citeturn15view0turn39view3  
- 因此工程上常见路线是“保持比较足够局部以控方差，但让局部信号比单 token 点估计更稳健”，如 top‑K 局部支持集匹配（truncated reverse‑KL）、熵引导 token 选择等。citeturn39view0turn28view0  

### 采样策略与批处理

常见采样旋钮包括：  
- **温度/Top‑p**：学生采样温度用于控制探索与多样性；如 GKD 在训练采样中使用 \(\gamma=1\) 以鼓励多样性。citeturn18view2  
- **多样本 rollout（pass@k）**：推理/评测时用多样本采样统计 pass@k 或 Avg@k；例如 ExOPD评测中对数学题每题采样 32 个解，对代码任务每题采样 4 个解，并设定最大生成长度 16384。citeturn36view2  
- **来自旧策略的 rollout + importance ratio**：当训练框架类似 PPO/GRPO，rollout 来自 \(\pi_{\theta_{\text{old}}}\)，更新用 \(\rho=\pi_\theta/\pi_{\theta_{\text{old}}}\) 进行校正；REOPOLD 的算法伪代码清晰体现了这一点。citeturn28view0  

### Mermaid：OPD 训练闭环流程图

```mermaid
flowchart TD
  A[从数据分布采样提示 x] --> B[学生策略 πθ 生成/采样序列 y ~ πθ(.|x)]
  B --> C[构造每步前缀 h_t=(x,y_<t)]
  C --> D[教师 π* 在 h_t 上计算 token 分布/评分信号]
  D --> E[计算蒸馏损失: Σ_t D(π*(.|h_t), πθ(.|h_t))]
  E --> F{是否加入额外项?}
  F -->|可选| G[加入RL/偏好奖励或KL约束到参考模型]
  F -->|仅蒸馏| H[仅用蒸馏损失]
  G --> I[stop-grad: 不对采样 y 回传, 仅对学生logits回传]
  H --> I
  I --> J[更新参数 θ ← θ - η∇θ L]
  J --> B
```

上述闭环与 GKD 的 on-policy KD / 混合采样框架一致：训练样本可在“固定数据轨迹”和“学生轨迹”间以比例 \(\lambda\) 切换；并采用不回传采样梯度的实现。citeturn18view0turn18view2  

**本节关键参考（2–4）**  
- GKD 的算法与 on-policy 损失、stop‑grad 说明（Algorithm 1 与式(4)）citeturn18view0turn18view1turn18view2  
- Revisiting OPD：序列级 reverse‑KL 到 token 级近似的偏差—方差分析与“sampled-token”脆弱性citeturn15view0turn39view3  
- REOPOLD：把对数似然比当作 token 奖励、并给出带 mask/裁剪/importance ratio 的训练算法citeturn28view0turn27view0  
- Veto：给出标准 OPKD token‑KL 形式与 logit 空间 PoE 桥接目标 \(Q\) 的公式化描述citeturn31view0  

## 关键变体与改进

本节按“目标函数/采样/稳定性/效率/信息形态”五个维度梳理 OPD 的主要变体，并给出代表性论文与年份。

### 发散度选择与温度调节

- **Forward KL vs Reverse KL vs JSD**：  
  - GKD 提供“可选 divergence + 可选轨迹来源”的统一视角，并在多个任务上报告 on-policy 变体普遍优于基线。citeturn18view2turn17view1  
  - OPSD 在推理任务上对 forward KL、reverse KL、JSD 做消融，并强调 divergence 选择与稳定性/性能相关。citeturn16view0  
  - Veto 进一步指出：forward KL 在“学生对教师偏好 token 概率接近 0”的早期阶段会出现梯度爆炸，而 reverse KL 易出现多样性/模式坍塌；并用单参数 \(\beta\) 统一地在两种 regime 下抑制病态梯度或控制“果断性—多样性”。citeturn31view0  

- **温度与采样多样性**：GKD 在训练生成中用 \(\gamma=1\) 鼓励学生采样多样性；而 REOPOLD 指出提高采样温度虽能增强探索，但可能引入更偏离教师的 token，从而放大对数似然比奖励方差，造成不稳定，这是“探索—对齐”张力的来源之一。citeturn18view2turn28view0  

### 数据来源混合与重要性采样

- **固定轨迹与 on-policy 轨迹混合（\(\lambda\)）**：GKD 的 Algorithm 1 以“student data fraction \(\lambda\)”在 on-policy 与数据集轨迹间抽样切换，统一 supervised KD 与 on-policy KD。citeturn18view0turn18view2  
- **教师混合采样（teacher-mixed sampling）**：MiniLLM 在优化 reverse KL 的策略梯度框架下提出 teacher-mixed sampling 等稳定技巧，用以缓解 reward hacking，并配合单步分解与长度归一化降低方差与长度偏置。citeturn35view0  
- **importance ratio 纠正**：当训练采用“旧策略 rollout + 新策略更新”，会引入 \(\rho\)；REOPOLD 的算法与 OVD 的 GRPO 训练都显式使用 importance ratio（并结合裁剪/正则）来稳定优化。citeturn28view0turn33view3  

### 基于价值/优势的加权与“OPD≈RL”泛化

- **G‑OPD/ExOPD：reward scaling / extrapolation**  
  G‑OPD把标准 OPD表述为 KL 约束 RL 的特例：奖励 \(r(x,y)=\log\frac{\pi^\*(y|x)}{\pi_{\text{ref}}(y|x)}\)，且奖励项与 KL 正则权重固定为 1:1；再通过引入 reward scaling \(\lambda\) 与可选参考模型 \(\pi_{\text{ref}}\) 形成更一般目标。其推导给出近似梯度形式：\(\nabla_\theta J=\mathbb{E}[\sum_t A_t \nabla_\theta\log\pi_\theta]\)，其中 \(A_t\) 由 teacher–student–ref 的对数概率差构成。citeturn19view3turn19view2  
  实验上作者报告当 \(\lambda>1\)（reward extrapolation）时，在多教师合并与强到弱蒸馏上可较标准 OPD进一步提升，并讨论了“reward correction”（用教师 pre‑RL base 作为 \(\pi_{\text{ref}}\)）的收益与额外计算成本。citeturn19view2turn36view2  

- **REOPOLD：放松 imitation 约束的“策略优化化”稳定技巧**  
  REOPOLD把 teacher–student 对数似然比作为 token 奖励，诊断其分布的“重尾负奖励”和“近零奖励退化”，并提出：  
  1) mixture-based reward floor（用 \(\log\frac{\lambda}{1-\lambda}\) 作为下界裁剪极端负奖励）；  
  2) 熵引导 token 级动态采样（只在高熵 token 上计算梯度）；  
  3) 探索→精炼两阶段训练。其 Algorithm 1 给出完整流程与关键超参（\(T_{\text{switch}},\lambda,\beta\) 等）。citeturn28view0turn27view0  

### 对抗性数据、前缀漂移与局部支持集匹配

- **sampled-token 近似的脆弱性与修复**：  
  “Revisiting OPD”指出常见 sampled-token OPD 把分布匹配退化为单 token 信号，尤其在长链推理/agentic 场景中会越来越不可靠；并给出三类失败模式：单 token 信号极不均衡、教师在学生前缀上不可靠、tokenizer/特殊 token 不匹配。citeturn15view0turn39view2  
  其提出 teacher top‑K 局部支持集匹配（truncated reverse‑KL）并配合 top‑p rollout、支持集重归一化与 special-token masking，报告在单任务数学与 ALFWorld+数学多任务训练中提升稳定性与指标。citeturn39view0turn39view2  

### 样本效率与显存/算力效率：从 logits 到“语言化评分”

- **全词表 logits vs sampled-token**：OPSD 的消融显示全词表 logit distillation 的效果优于 sampled-token distillation。citeturn16view0  
- **OVD：用离散 verbal score（0–9）替代 logits**：OVD指出 token‑logit 监督在长序列与大词表下有严重显存瓶颈，并给出具体量化：例如在序列长度 8192、词表 152K 时，单批次保存 logits（FP32）就达 GB 级并随长度线性增长；因此提出用轨迹级“语言化评分+拒绝采样+GRPO”进行 on-policy 蒸馏，以提升记忆效率并支持 black-box 教师。citeturn32view0turn33view0  

**本节关键参考（2–4）**  
- MiniLLM（reverse KL 的策略梯度优化、单步分解/teacher-mixed/长度归一化等稳定技巧，ICLR 2024）citeturn35view0  
- REOPOLD（奖励裁剪+熵引导动态采样+两阶段训练，2026）citeturn28view0turn27view0  
- Revisiting OPD（top‑K 局部支持集匹配、mask、top‑p rollout 等，2026）citeturn39view0turn39view2  
- OVD（用 verbal score 降显存并支持 black-box 蒸馏，2026）citeturn33view0turn32view0  

## 实验与性能证据

### 常用基准与评价指标

OPD 在近两年主要被用于三类任务，并对应不同评测指标：  
- **生成质量任务**：摘要（XSum）常用 ROUGE‑2/ROUGE‑L；翻译常用 BLEU；这些是 GKD 的主要实验场景之一。citeturn17view1turn18view0  
- **推理/代码任务**：数学竞赛题（AIME、HMMT、Math500、Minerva、OlympiadBench 等）多用准确率或 Avg@k/pass@k；代码生成用 HumanEval+/MBPP+/LiveCodeBench 并用 pass@k。citeturn39view0turn36view2turn16view0  
- **检索/交互式 QA 与 agent 场景**：Web Q&A 常用 EM（exact match）；多轮环境（如 ALFWorld）用任务成功率；OVD 报告在 Web Q&A 上对多组强基线的 EM 提升。citeturn33view3turn39view0  

### 典型实验设置与超参模式（归纳）

跨论文可观察到较一致的工程化配置模式：  
- **推理类 OPD 经常采用多样本采样评测**（pass@k/Avg@k），以减轻单次采样方差；例如 ExOPD评测数学每题 32 解、代码每题 4 解，并设置较长最大生成长度。citeturn36view2  
- **短预算、低步数快速收敛的趋势**：OPSD 报告仅 100 steps、每题 1 rollout、每题 1024 sampled tokens 即可获得明显收益，并给出与 GRPO 的 token 效率对比；并公开训练配置（LoRA、effective batch size、学习率等）。citeturn16view0turn16view3  
- **稳定性控制项逐渐变得“像 RL”**：mask、KL 正则、reward/ratio 裁剪、熵/不确定性引导采样等，在 REOPOLD、OVD、Revisiting OPD 中都以不同形式出现。citeturn28view0turn33view3turn39view0  

### 代表性结果比较表（方法/论文/基准/结果/备注）

| 论文/年份 | 方法 | 基准 | 结果（论文中报告的代表性结论/数字） | 备注 |
|---|---|---|---|---|
| ICLR 2024 On-Policy Distillation of Language Models（GKD） | GKD：混合数据+自生成轨迹、可选 divergence；stop‑grad | XSum（ROUGE‑2）、算术推理（GSM8K）、等 | 报告 on-policy GKD 变体普遍优于多种 KD 基线；并指出在 XSum 上，**仅用 5% 子集且无真值摘要的 on-policy 训练**可优于使用全量真值的 supervised KD/ImitKD。citeturn17view1turn17view2 | 提供 Algorithm 1 与 on-policy loss；强调不回传采样梯度以稳定/高效。citeturn18view0turn18view2 |
| ICLR 2024 MiniLLM | reverse KL 蒸馏 + 策略梯度优化；单步分解/teacher-mixed/长度归一化 | 指令跟随 KD（多数据集，ROUGE‑L、人评、GPT‑4 反馈等） | 报告 MiniLLM 在多模型族（120M–13B）上优于 SeqKD/标准 KD，并给出“更低 exposure bias、更好校准、更强长文本生成”等分析；并提供代码与检查点。citeturn35view0 | 代表“reverse‑KL on-policy 优化”路线；显式引用 policy gradient。citeturn35view0 |
| 2026 Revisiting OPD | teacher top‑K 局部支持集匹配（truncated reverse‑KL）+ top‑p rollout + masking | 单任务数学（Math500/AIME24/AIME25/Minerva/OlympiadBench）与多任务（ALFWorld+数学） | 单任务数学：sampled-token OPD 平均 36.4；加 mask 40.7；作者方法 41.5。citeturn39view0 多任务：在保持 ALFWorld 强性能的同时，数学侧显著提升（如 Math500 76.0→82.0），平均数学分提升 36.6→41.7。citeturn39view2 | 系统总结 sampled-token 失败模式：单 token 信号不均衡、教师前缀不可靠、tokenizer/特殊 token 不匹配。citeturn15view0turn39view2 |
| 2026 G‑OPD / ExOPD | OPD≈KL 约束 RL；引入 reward scaling \(\lambda\) 与参考模型；ExOPD=\(\lambda>1\) | 数学（AIME24/25、HMMT25 Feb/Nov）；代码（HumanEval+/MBPP+/LiveCodeBench） | 给出统一评测设置（温度 1.0、top‑p 1.0、maxlen 16384；数学每题 32 解、代码每题 4 解）。citeturn36view2 并报告 reward extrapolation 在多设置中优于标准 OPD，\(\lambda=1.25\) 时在多基准上可超过 teacher/OPD（论文叙述）。citeturn36view2turn19view1 | 指出 \(\lambda\neq 1\) 需额外计算 \(\log \pi_{\text{ref}}\)，并讨论用 teacher pre‑RL base 作为 ref 的“reward correction”会增加算力。citeturn19view2 |
| 2026 OPSD（On-Policy Self‑Distillation） | 单模型双角色：teacher 端给“特权信息”（真值/推理链），student 端仅见问题；token 级 divergence；仅对 student logits 反传 | 数学竞赛基准（AIME24/25/HMMT25） | Table 2：Qwen3‑1.7B 平均从 37.1（base）→ 43.4（OPSD）；并在 100 steps 内收敛，且每题仅 1 rollout、1024 token；对比 GRPO 显示更高 token 效率。citeturn16view0turn16view1 | 解释为“密集 token 级奖励”的 policy gradient；并提供代码仓库。citeturn16view0turn16view3 |
| 2026 OPCD（On‑Policy Context Distillation） | 学生无上下文生成；teacher 有上下文；最小化 reverse KL；用于经验知识/系统提示内化 | OOD 评测（IF‑Eval 等）、安全/医疗跨域 | 报告 OPCD 相对 off-policy context distill 在 Frozen Lake 上 IF‑Eval 提升约 2%，并在安全→医疗 OOD 上较 off-policy baseline 高约 4 点，缓解遗忘。citeturn34view0 | 讨论 teacher‑student 配置比 self‑OPCD 更稳定，并提到 EMA teacher 可缓解自蒸馏不稳。citeturn34view0 |
| 2026 OVD（On‑policy Verbal Distillation） | 用 teacher 的离散 verbal score（0–9）做轨迹评价+拒绝采样+GRPO；避免全词表 logits 存储 | Web Q&A（8 数据集，EM）；数学推理 | 报告 Web Q&A 平均 EM 最高可 +12.9%，数学基准在“仅 1 随机样本训练”时最高 +25.7%。citeturn33view0turn33view1 并给出具体对比：如 Qwen2.5‑3B‑Base 上平均 43.6% vs Search‑R1 32.8%。citeturn33view3 | 量化 token‑logit 蒸馏显存瓶颈（例：8192×152K logits 等），强调黑盒兼容与 memory‑efficient。citeturn32view0 |
| 2026 REOPOLD | 放松 OPD：mixture‑based reward floor + 熵引导动态采样 + 两阶段训练 | AIME‑25、视觉推理等 | 报告相对 RL 基线 6.7–12× 样本效率提升，并称 7B 学生可在视觉推理上匹配 32B 教师且推理速度约 3.32×。citeturn27view0 | 对“重尾负奖励/近零奖励退化/熵塌缩”给出诊断与 Algorithm 1。citeturn28view0 |

### 局限性与可复现实验注意点

尽管大量结果显示 OPD 在若干任务上具备优势，但文献也强调其局限：  
- **目标与代理指标错位**：teacher matching 并不等价于任务成功；局部目标仍可能是“截断代理”。citeturn39view2  
- **教师在学生前缀上的不可靠性**：当学生滚动到教师少见的前缀，token 级监督可能误导；需要局部支持集、mask、以及对 teacher 不确定性/分布外前缀的处理。citeturn15view0turn39view2  
- **早期训练的梯度病态与多样性坍塌**：Veto 明确指出 forward KL 的梯度爆炸与 reverse KL 的模式坍塌风险，并将其归因为 divergence 几何而非数据策略；这意味着“改采样”不足以完全解决，必须动目标函数。citeturn31view0  

**本节关键参考（2–4）**  
- GKD：跨任务的 on-policy 蒸馏优势与数据效率结论citeturn17view1turn17view2  
- Revisiting OPD：table 结果与 failure modes/修复策略citeturn39view0turn39view2  
- OPSD：推理基准、token 效率与完整训练配置表citeturn16view0turn16view3  
- OVD：Web Q&A/数学基准增益与显存瓶颈量化citeturn33view0turn32view0  

## 应用场景与工程注意事项

### 典型应用场景

1) **强到弱能力迁移（Reasoning/Code）**：以更大教师或更强后训练教师（如 RL‑Math/RL‑Code 版本）指导小模型，在数学/代码任务上提升 pass@k/准确率；ExOPD、REOPOLD、OPSD 都以此为核心叙事之一。citeturn36view2turn27view0turn16view0  

2) **多教师能力合并（capability merging）**：从多个“领域专家教师”回灌到统一学生。G‑OPD/ExOPD 特别讨论了把多个 domain RL variant 合并回 base 学生并可能超过单一教师边界的设定。citeturn6view1turn19view1  

3) **上下文/系统提示内化（从 in‑context 到参数）**：OPCD 把“上下文蒸馏”与 on-policy 结合，用学生轨迹做 reverse KL，让学生在无上下文条件下复现 teacher（有上下文）行为，并强调对 OOD 遗忘更友好。citeturn34view0turn23search15  

4) **agentic 训练与交互式任务**：在多轮工具使用/环境交互中，sampled-token 目标更脆弱；Revisiting OPD 在 ALFWorld+数学混合训练中展示局部支持集匹配的收益。citeturn39view2  

5) **黑盒教师与资源受限训练**：当无法获得 teacher logits 或 logits 存储过贵时，OVD用 verbal score 做轨迹监督以降低显存，并声称仍能带来显著增益。citeturn33view0turn32view0  

### 计算成本与系统实现要点

- **教师前向的主导成本**：OPD的瓶颈通常不是“生成学生样本”，而是“对每个前缀求 teacher 分布”。GKD指出在无标注 prompts 上，用学生生成序列比教师更便宜，但仍需要教师在这些序列上提供 logits 监督。citeturn18view2  
- **全词表 logits 的显存爆炸**：OVD给出定量例子说明在长序列/大词表下存 logits 远超加速器容量（例如 8192×152K 的 logits 组件达到 GB 级且线性增长），这直接推动了 sampled-token、top‑K 截断、或 verbal distillation 等近似。citeturn32view0  
- **“近似—稳定性”需要共同设计**：Revisiting OPD 表明 sampled-token 虽便宜但脆弱；其 top‑K 局部支持集匹配在保留“局部、相对便宜”的同时显著提升稳定性与结果。citeturn15view0turn39view0  
- **参考模型与额外对数概率的开销**：G‑OPD指出当 reward scaling \(\lambda\neq 1\) 时需额外计算 \(\log \pi_{\text{ref}}\)；若 \(\pi_{\text{ref}}\) 选择更大模型（如 teacher pre‑RL），会进一步增加成本。citeturn19view2  
- **并行/分布式采样**：当采用 PPO/GRPO 类框架，rollout 端与 learner 端往往分离；REOPOLD/OVD的算法形式天然兼容“多 worker 采样、中心 learner 更新”的 pipeline，但会带来 on-policy 滞后与 ratio 校正需求。citeturn28view0turn33view3  

### 在线数据收集的隐私与安全风险

OPD 及其在线延伸（如 OPCD 的在线经验循环）往往依赖交互数据、用户侧轨迹或系统日志，这会引入三类风险：

- **隐私泄露与模型记忆**：大模型可能记忆训练样本，攻击者可通过查询恢复训练文本片段。训练数据提取攻击在公开工作中被系统展示，并指出更大模型更易受影响。citeturn40search2turn40search10  
- **训练数据投毒与供应链风险**：在线收集与自动化数据管线更易被投毒；OWASP 的 LLM 风险列表把 Training Data Poisoning、Supply Chain 等列为关键风险类别。citeturn40search1  
- **合成数据反馈回路（自生成数据占比升高）导致能力退化**：当训练越来越依赖模型生成数据，存在“model collapse”类风险；相关研究在生成模型递归训练下观察到分布尾部消失与性能劣化，并给出理论/实证分析。citeturn40search11turn40search15  

工程建议（与文献结论一致的“保守做法”）：  
1) 明确区分“训练数据（可持久化）”与“短期评估缓存”，对敏感信息做最小化采集与脱敏；  
2) 为在线管线配置数据 provenance、审计与投毒检测；  
3) 控制模型自生成数据的占比、保留真实数据锚点；  
4) 在使用 black-box 教师/环境代理时，对评分/反馈的鲁棒性与攻击面（prompt injection、工具滥用）做红队测试。citeturn40search0turn40search1turn40search11  

**本节关键参考（2–4）**  
- OVD：长序列 token‑logit 监督的显存瓶颈量化与 verbal distillation 的动机citeturn32view0turn33view0  
- G‑OPD：参考模型选择与额外计算成本、以及 OPD 与 KL‑RL 的关系citeturn19view2turn19view3  
- NIST AI RMF（风险管理框架，含可信/安全/治理视角）citeturn40search0turn40search12  
- Carlini 等：训练数据提取攻击与隐私风险证据citeturn40search2turn40search10  

## 开放问题与研究方向

### 理论问题：收敛性、偏差—方差、样本复杂度

- **OPD 的“真目标”与“可训练代理目标”差距**：sampled-token、truncated support、mask、Veto/PoE bridge 等都在改变优化的实际目标。如何在理论上刻画这些代理目标与“期望行为提升”的关系，以及在什么条件下保证收敛/不退化，仍是开放问题。citeturn15view0turn31view0  
- **on-policy 蒸馏更新是否为梯度场**：在 RL 语境下，student-driven 的一步蒸馏更新可能不形成梯度向量场，并在与奖励混合时出现非收敛动力学；AISTATS 2019 的系统分析提示需要更严格的目标构造（如期望熵正则蒸馏）以获得保证。citeturn29view0  
- **偏差—方差的可控插值**：Revisiting OPD 的分析提出“更强未来回报耦合→更高梯度方差”的规律；如何构造“信息量足够、方差可控”的中间形式（折扣回报、支持集大小、熵阈值等），是当下大量方法的共同主题。citeturn15view0turn28view0  

### 方法问题：teacher 不确定性、分布外前缀与长链稳定性

- **teacher 在学生前缀上的可靠性建模**：当学生走到 teacher 罕见前缀，teacher logits 可能并非“好指导”。现有工作多用启发式（top‑K 支持、mask、PoE bridge、熵过滤）；更系统的 teacher 不确定性估计、以及“何时信 teacher、何时自主探索”的策略仍待深入。citeturn39view2turn31view0turn28view0  
- **多教师合并的冲突与权衡**：ExOPD 报告 reward extrapolation 可在多教师合并中带来收益，但也伴随响应长度/熵变化等行为漂移；如何在“合并能力”与“保持通用对齐/风格”间给出可解释的控制机制（类似预算控制推理）仍是方向。citeturn19view1turn36view2  

### 与人类反馈结合的可能性

- **OPD 与 RLHF/RLAIF 的组合**：GKD明确给出把 on-policy distillation loss 与标量奖励结合的正则化 RL 目标形式，提示 OPD 可作为“贴近教师/参考策略”的结构化约束项。citeturn17view2turn17view3  
- **从“token 级一致”到“偏好/可验证过程”**：OVD把教师信号从 logits 变为 step 级 verbal score，OPSD把 teacher 的“特权信息”嵌入上下文，OPCD把系统提示与经验知识内化；这些都暗示“人类反馈/验证器/系统提示”可以通过 OPD 变体变成更密集、更可学习的信号。citeturn16view1turn34view0turn33view0  

### 安全与持续学习：避免自强化与模型退化

在线/持续 OPD（如在线经验循环）必须面对：  
- **数据投毒、prompt injection、工具滥用**等应用层攻击面（OWASP 列表持续更新并强调其跨生命周期风险）。citeturn40search1turn40search5  
- **合成数据递归训练导致的 model collapse**：当“学生自生成数据”在训练集中占比过高，可能出现尾部分布消失与能力退化，要求持续学习系统保留真实数据锚点与审计机制。citeturn40search11turn40search15  

**本节关键参考（2–4）**  
- Distilling Policy Distillation（AISTATS 2019：收敛性/梯度场问题与期望熵正则蒸馏）citeturn29view0  
- Revisiting OPD（token 级与序列级目标的偏差—方差权衡、长链 failure modes）citeturn15view0turn39view0  
- GKD（OPD 与 RLHF/RLAIF 结合的正则化目标形式）citeturn17view2turn17view3  
- Shumailov 等关于 model collapse 的证据与理论分析线索（Nature 2024 及后续统计分析）citeturn40search11turn40search15  

## 主要参考文献（不要求完整书目）

- On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes（ICLR 2024，GKD/OPD 框架与算法）citeturn22search9turn18view0turn18view2  
- MiniLLM: Knowledge Distillation of Large Language Models（ICLR 2024，reverse KL + policy gradient 的蒸馏路线，含代码）citeturn35view0turn22search1  
- Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes（2026，top‑K 支持集匹配与稳定性分析）citeturn15view0turn39view0  
- Stable On-Policy Distillation through Adaptive Target Reformulation（Veto，2026，logit 空间桥接目标）citeturn31view0  
- Scaling Reasoning Efficiently via Relaxed On-Policy Distillation（REOPOLD，2026，奖励裁剪+熵引导采样+两阶段训练）citeturn27view0turn28view0  
- Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation（G‑OPD/ExOPD，2026，OPD≈KL‑RL、reward scaling）citeturn19view3turn36view2  
- Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models（OPSD，2026，特权信息自蒸馏）citeturn16view1turn16view0turn16view3  
- On-Policy Context Distillation for Language Models（OPCD，2026，把 in‑context 经验/系统提示内化进参数）citeturn34view0turn23search15  
- OVD: On-policy Verbal Distillation（2026，verbal score 轨迹匹配，降低显存并支持 black-box）citeturn32view0turn33view0  
- Distilling Policy Distillation（AISTATS 2019，策略蒸馏谱系与收敛性讨论）citeturn29view0  
- A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning（DAgger，模仿学习 on-policy 数据聚合）citeturn2search0  
- Artificial Intelligence Risk Management Framework（NIST AI RMF 1.0，治理/可信风险框架）citeturn40search0turn40search12  
- OWASP Top 10 for Large Language Model Applications（LLM 安全风险分类，如数据投毒等）citeturn40search1turn40search5  
- Extracting Training Data from Large Language Models（训练数据提取攻击，隐私风险证据）citeturn40search2turn40search10  
- AI models collapse when trained on recursively generated data（Nature 2024，合成数据递归训练导致退化）citeturn40search11