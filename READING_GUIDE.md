# Nano-vLLM 阅读路线指引

## 先看哪里（5分钟建立全局）
1. `example.py`
- 看最小使用方式：`LLM(...)` + `SamplingParams(...)` + `generate(...)`。
- 明确这个项目对外 API 基本对齐 vLLM。

2. `nanovllm/llm.py` 和 `nanovllm/__init__.py`
- `LLM` 只是 `LLMEngine` 的薄封装，真正逻辑全在引擎层。

## 主阅读路径（按调用链）
1. `nanovllm/engine/llm_engine.py`
- 入口调度器：负责请求接入、批处理循环、吞吐统计、结果回收。
- 关键特性：
  - 多请求批处理（prefill/decode 两阶段）
  - Tensor Parallel 多进程启动（`torch.multiprocessing`）
  - 对外统一 `generate` 接口

2. `nanovllm/engine/scheduler.py`
- 批处理核心策略：先 prefill，后 decode；内存不足时抢占（preempt）。
- 关键特性：
  - `waiting/running` 双队列调度
  - `max_num_seqs`、`max_num_batched_tokens` 约束
  - EOS / `max_tokens` 完成判定

3. `nanovllm/engine/block_manager.py`
- KV Cache 块管理器。
- 关键特性：
  - 块分配/回收（free/used block）
  - 基于哈希的 Prefix Cache 复用（`xxhash`）
  - decode 阶段按 block 追加

4. `nanovllm/engine/model_runner.py`
- 真正执行模型前向的“重模块”。
- 关键特性：
  - NCCL 初始化与 TP rank 协作
  - warmup 后按显存动态分配 KV Cache
  - prefill/decode 输入打包（`slot_mapping`、`block_tables`）
  - CUDA Graph 捕获与回放（可关闭）
  - 采样调用与进程间共享内存通信

## 模型与算子层（理解性能关键）
1. `nanovllm/models/qwen3.py`
- Qwen3 的完整解码器实现（Attention + MLP + RMSNorm + LM Head）。
- 同时定义了权重“打包映射”，供加载器把 HF 权重映射到并行层。

2. `nanovllm/layers/attention.py`
- FlashAttention 路径（varlen prefill + kvcache decode）。
- Triton kernel 负责把 K/V 写入缓存。

3. `nanovllm/layers/linear.py`、`embed_head.py`
- Tensor Parallel 线性层和词表并行 embedding/lm head。
- 重点看各 `weight_loader` 的切分逻辑（决定 TP 正确性）。

4. `nanovllm/layers/rotary_embedding.py`、`sampler.py`、`layernorm.py`
- RoPE、采样、RMSNorm；都做了 `torch.compile`，是推理热路径。

## 配置与辅助模块
- `nanovllm/config.py`：运行参数入口（批大小、上下文长度、TP、显存利用率）。
- `nanovllm/sampling_params.py`：采样温度、长度、EOS 行为。
- `nanovllm/utils/context.py`：全局上下文（prefill/decode 的元数据桥接）。
- `nanovllm/utils/loader.py`：safetensors 权重加载与 packed 权重拆分。

## 建议阅读策略
- 第一轮：只追一条请求从 `generate -> schedule -> run -> postprocess` 的生命周期。
- 第二轮：只盯 KV Cache（`BlockManager` + `Attention` + `Context`）。
- 第三轮：只盯 TP（`ModelRunner` + `linear/embed_head` 的分片与通信）。

这三轮看完，你就能判断这个项目的吞吐瓶颈和可改造点。
