# Chinese classical text generation

from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-ancient\snapshots\3b264872995b09b5d9873e458f3d03a221c00669")
tokenizer = BertTokenizer.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-ancient\snapshots\3b264872995b09b5d9873e458f3d03a221c00669")
print(model)

# Use Pipeline to call the model
text_generator = TextGenerationPipeline(model,tokenizer,device="cuda")

# Use text_generator to generate text
# do_sample determines whether to perform random sampling. 
# If True, the result is different each time; if False, the result is the same each time.
for i in range(3):
    print(text_generator("于是者", max_length=100, do_sample=True))