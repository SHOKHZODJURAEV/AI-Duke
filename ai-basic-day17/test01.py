import lmdeploy
from lmdeploy import pipeline, PytorchEngineConfig
pipe = pipeline('/root/autodl-tmp/llm/Qwen/Qwen1.5-1.8B-Chat',
                backend_config=PytorchEngineConfig(
                    max_batch_size=32,
                    enable_prefix_caching=True,
                    cache_max_entry_count=0.8,
                    session_len=8192,
                ))
response = pipe(["Hi, pls intro yourself", "Shanghai is"])
print(response)