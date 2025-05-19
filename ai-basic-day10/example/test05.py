# Chinese Poetry
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-poem\snapshots\6335c88ef6a3362dcdf2e988577b7bafeda6052b")
tokenizer = BertTokenizer.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-poem\snapshots\6335c88ef6a3362dcdf2e988577b7bafeda6052b")
print(model)

# Use Pipeline to call the model
text_generator = TextGenerationPipeline(model, tokenizer, device="cuda")

# Use text_generator to generate text
# do_sample determines whether to perform random sampling. If True, the result is different each time; if False, the result is the same each time.
for i in range(3):
    print(text_generator("[CLS]白日依山尽，", max_length=50, do_sample=True))