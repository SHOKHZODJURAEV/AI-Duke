from transformers import AutoModelForCausalLM,AutoTokenizer,TextGenerationPipeline
import torch

tokenizer = AutoTokenizer.from_pretrained(r"D:\PycharmProjects\day_18demo\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
model = AutoModelForCausalLM.from_pretrained(r"D:\PycharmProjects\day_18demo\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")

#  (Chinese poetry)
model.load_state_dict(torch.load("net.pt"))

# Use the built-in pipeline tool to generate content
pipline = TextGenerationPipeline(model,tokenizer,device=0)

print(pipline("天高", max_length=24))