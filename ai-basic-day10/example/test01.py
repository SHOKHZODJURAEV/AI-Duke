# Chinese colloquial text generation
from transformers import GPT2LMHeadModel,BertTokenizer,TextGenerationPipeline

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
tokenizer = BertTokenizer.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
print(model)

# Use Pipeline to call the model
text_generator = TextGenerationPipeline(model,tokenizer,device="cuda")

# Use text_generator to generate text
# do_sample determines whether to perform random sampling. If True, the result is different each time; if False, the result is the same each time.
for i in range(3):
    print(text_generator("这是很久之前的事情了,", max_length=100, do_sample=True))