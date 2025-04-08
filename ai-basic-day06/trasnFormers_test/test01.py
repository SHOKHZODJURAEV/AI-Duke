# Download the model locally
from transformers import AutoModelForCausalLM,AutoTokenizer

# Download the model and tokenizer locally and specify the save path
model_name = "uer/gpt2-chinese-cluecorpussmall"
cache_dir = "model/uer/gpt2-chinese-cluecorpussmall"

# Download the model
AutoModelForCausalLM.from_pretrained(model_name,cache_dir=cache_dir)
# Download the tokenizer
AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)

print(f"The model and tokenizer have been downloaded to: {cache_dir}")