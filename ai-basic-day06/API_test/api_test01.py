import requests

API_URL = "https://api-inference.huggingface.co/models/uer/gpt2-chinese-cluecorpussmall"

# Anonymous access without using a token
response = requests.post(API_URL, json={"inputs": "你好，Hugging face"})
print(response.json())