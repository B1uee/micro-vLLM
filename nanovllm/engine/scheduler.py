from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

"""
模块机制（nano-vLLM）：
- 使用 waiting/running 双队列。
- 调度分两段：有可接纳请求时优先 prefill，否则进入 decode。
- 资源不足时对 running 做抢占（preempt），回退到 waiting。

与 vLLM v1（Scheduler）主要差异：
- 本模块是“显式 prefill/decode 两阶段”调度。
- vLLM v1 的公开实现是“统一 token 预算驱动”的调度思路，
  不把 prefill/decode 作为硬分段概念，便于覆盖 chunked prefill、
  speculative decode 等更通用场景。
- 本模块目前是轻量 FCFS 风格，能力简单但易读。
"""

class Scheduler:
    """
        总结：
        1. 有新请求能 prefill 就优先 prefill。
        2. decode 阶段按资源压力可抢占，避免死锁。
        3. running 队列顺序被维护，用于下一轮继续推进。

    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # Prefill 阶段：在 token/block 预算内尽量接纳 waiting 请求。
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 约束：token数量和可用block
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq) #  分配/复用 block
            num_batched_tokens += len(seq) - seq.num_cached_tokens # 扣除 prefix cache 命中部分
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft() 
            self.running.append(seq)
            scheduled_seqs.append(seq) # 从 waiting 挪到 running，并加入 scheduled_seqs
        if scheduled_seqs:
            return scheduled_seqs, True # 只要 prefill 收到了至少一个序列，立即返回：(scheduled_seqs, True)

        # Decode 阶段：每个 running 序列每轮解 1 个 token，资源紧张时触发抢占。
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 若还有其他 running，优先抢占队尾那个（最新进入 running 的)
                    self.preempt(self.running.pop())
                else:
                    # 否则连当前 seq 自己也抢占回 waiting
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq) # 新开尾块或封存满块
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs)) # 按原顺序塞回 running 头部
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # 回收其 KV 所有权并放回 waiting 队列。
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # 遇到 EOS（除非忽略）或达到最大生成长度则结束。
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
