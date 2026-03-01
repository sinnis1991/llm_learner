# SGLang PD 分离架构笔记：Prefill / Decode / Router 协同机制

- 日期：2026-03-02
- 来源：微信公众号「jacky的技术课堂」
- 原文链接：https://mp.weixin.qq.com/s/RYQ547Gxktza0_PPhfn_Iw
- 原文标题：SGLang PD分离架构深度解析：Prefill、Decode与Router的协同之道

---

## 一、核心结论

这篇文章的主线是：**把 Prefill 和 Decode 从同一节点解耦出来**，能显著提升推理吞吐与资源利用率。

相比传统“同机混跑”架构，PD 分离的收益在于：

1. Prefill（计算密集）和 Decode（内存/带宽密集）不再抢同一类资源。  
2. 硬件可按角色优化（Prefill 偏算力，Decode 偏大显存/带宽）。  
3. 可独立扩缩容，系统弹性更好。  
4. 通过 Router + 直连 KV 传输，避免 Router 成为数据平面瓶颈。

---

## 二、三角色分工（SGLang）

### 1) Router（控制平面）

主要职责：

- 选择 Prefill/Decode 节点对（pair）
- 注入 bootstrap 三元组
- 转发请求并聚合 Decode 的流式输出

关键点：

- 支持多种路由策略（random / power_of_two / cache_aware / bucket）
- 控制平面与数据平面分离：Router 不承载 KV 大数据传输

### 2) Prefill Worker（KV 生产者）

主要职责：

- 执行 prefill forward
- 生成 KV cache
- 将 KV 分块异步传输给 Decode

实现上使用三阶段队列（Bootstrap / Waiting / Inflight）管理请求生命周期。

### 3) Decode Worker（Token 生成者）

主要职责：

- 按 bootstrap 信息连接 Prefill 侧服务
- 接收并提交 KV cache
- 以 prebuilt batch 方式跳过 prefill，直接 decode 生成 token

---

## 三、我认为最关键的机制

### 1) Bootstrap 三元组

Router 给请求注入：

- `bootstrap_host`
- `bootstrap_port`
- `bootstrap_room`

作用是让 Decode 精准连到对应 Prefill 的“房间”，完成 KV 通道建立。

### 2) 异步流水线

Prefill 在传输上一请求 KV 的同时可以继续处理下一请求，形成流水线，显著提高 GPU 持续利用率。

### 3) Prebuilt Batch

Decode 侧拿到 KV 后构造 `ForwardMode.PREBUILT`，跳过 prefill 计算，直接进入 token 生成循环。

### 4) 控制/数据解耦

- 控制平面：Router 负责调度与编排
- 数据平面：Prefill ↔ Decode 直连传输 KV（如 RDMA/Mooncake）

这是规模化落地的关键工程点。

---

## 四、部署与调优要点（提炼）

1. Decode 常是瓶颈，通常需要更多 Decode worker。  
2. Prefill 可配高算力卡，Decode 优先大显存与带宽。  
3. 路由策略可分角色配置：Prefill 偏 cache-aware，Decode 偏负载均衡。  
4. 重点监控：prefill/decode 时延、KV 传输时延、tokens/s、worker 负载、cache 命中率。

---

## 五、个人启发

1. 在大模型服务里，**架构层面优化（分离+流水线）常比单点 kernel 优化收益更稳定**。  
2. Router 不应承载大数据通道，否则扩展性会被网关层卡死。  
3. 对长上下文/高并发场景，PD 分离是可落地且可解释的工程方案。  
4. 后续可重点跟进：KV 压缩策略、分块粒度、路由策略自适应。
