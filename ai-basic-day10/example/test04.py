# Chinese couplets

from transformers import BertTokenizer,GPT2LMHeadModel,TextGenerationPipeline

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-couplet\snapshots\91b9465fb1be617f69c6f003b0bd6e6642537bec")
tokenizer = BertTokenizer.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-couplet\snapshots\91b9465fb1be617f69c6f003b0bd6e6642537bec")
print(model)

# Use Pipeline to call the model
text_generator = TextGenerationPipeline(model,tokenizer,device="cuda")

# Use text_generator to generate text
# do_sample determines whether to perform random sampling. 
# If True, the result will be different each time; 
# if False, the result will be the same each time.
for i in range(3):
    print(text_generator("[CLS]十口心思，思想思国思社稷", max_length=28, do_sample=True))