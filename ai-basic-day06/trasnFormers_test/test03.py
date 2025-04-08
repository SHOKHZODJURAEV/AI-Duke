from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load the model and tokenizer
model_name = r"/Users/shokhzodjuraev/Desktop/AI-Duke/AI-BasicStart/AI-Duke/ai-basic-day06/trasnFormers_test/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Create a classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer,device="cuda")

# Perform classification
result = classifier("你好，我是一款语言模型")
print(result)
print(model)


