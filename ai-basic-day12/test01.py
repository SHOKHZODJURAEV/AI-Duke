# Model download
from modelscope import snapshot_download
model_dir = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct',cache_dir="/teacher_data/zhangyang/llm/")