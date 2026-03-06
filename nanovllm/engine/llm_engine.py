import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 仅透传 Config 已定义字段；其余参数忽略。
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = [] # TP 子进程对象
        self.events = []
        ctx = mp.get_context("spawn")
        # Rank 0 留在当前进程，其余 TP rank 作为子进程运行。
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # “启动一个 TP 子进程，并让它一启动就执行ModelRunner(config, i, event)”。
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events) # rank 0,额外承担了调度/采样/返回结果的控制开销。
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        # 先广播 exit 给 TP 子进程，再 join 回收。
        # self.model_runner 是一个对象引用，如果没有其他引用了，Python 会回收它
        # 所以先 exit() 让对象变成可回收状态
        self.model_runner.call("exit")
        del self.model_runner 
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # 由调度器决定本轮是 prefill 还是 decode。
        seqs, is_prefill = self.scheduler.schedule() # -> tuple[list[Sequence], bool]
        token_ids = self.model_runner.call("run", seqs, is_prefill) # 所有 TP rank 一起跑这一轮，返回每个序列新生成的 token
        self.scheduler.postprocess(seqs, token_ids) # 把 token 追加回对应序列，从 running 移除完成的request并回收资源
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished] # 提取已完成输出 + 统计吞吐口径
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs) # 单变量复用，节省一个状态变量
        # outputs: 本轮刚完成的序列结果（可能为空，也可能多个）
        # num_tokens: 给外层 generate() 用来算 prefill/decode 吞吐
        return outputs, num_tokens
    

    def is_finished(self):
        return self.scheduler.is_finished()

    # scheduler是continus batching风格，但外部的generate()是离线批处理，等待request全部完成才返回结果
    def generate(
        self,
        prompts: list[str] | list[list[int]],  # 文本 or token ids
        sampling_params: SamplingParams | list[SamplingParams],  # 采样策略(单个对象：会自动复制给所有 prompt；列表，需要与prompts一一对应)
        use_tqdm: bool = True,
    ) -> list[str]:
        # 将单个 sampling 参数展开为逐请求参数列表。
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            # 复制
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp) # zip一一对应
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished(): # 直到所有request都结束
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                # num_tokens > 0 表示 prefill；< 0 表示 decode。
                # 参考：num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        # 按 seq_id 排序，保证返回顺序与输入一致
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 把每个 token_ids decode 成文本并打包，此时可以丢掉seq_id
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
