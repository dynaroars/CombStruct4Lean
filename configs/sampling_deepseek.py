from prover.utils import AttrDict
from prover.algorithms import Sampling


# dataset
data_path = "datasets/benchmark_dsprover.jsonl"
data_split = "test"
data_repeat = 1

# verifier
lean_max_concurrent_requests = 8
lean_memory_limit = 10
lean_timeout = 300

# model
batch_size = 64
model_path = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
quantized_model_path = None
model_args = AttrDict(
    mode="cot",  # `cot` or `non-cot`
    temperature=1,
    max_tokens=2048,
    top_p=0.95,
)

# algorithm
n_search_procs = 64
sampler = dict(
    algorithm=Sampling,
    sample_num=10,
    log_interval=32,
)
