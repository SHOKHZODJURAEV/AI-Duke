from transformers import AutoModelForCausalLM,AutoTokenizer,TextGenerationPipeline
import torch

tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Mr.Zhang\.cache\huggingface\hub\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
model = AutoModelForCausalLM.from_pretrained(r"C:\Users\Mr.Zhang\.cache\huggingface\hub\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")

# Load our own trained weights (Chinese ancient poetry)
model.load_state_dict(torch.load("net.pt"))

# Use the built-in pipeline tool to generate content
pipeline = TextGenerationPipeline(model,tokenizer,device=0)

print(pipeline("天高",max_length =24))