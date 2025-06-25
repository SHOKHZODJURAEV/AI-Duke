# Use transformers to load the llama3 model
from transformers import AutoModelForCausalLM,AutoTokenizer

DEVICE = "cuda"
# Load the local model, the path is the root directory of the model configuration file
model_dir = "/teacher_data/zhangyang/llm/LLM-Research/Llama-3___2-1B-Instruct/"
# Use transformers to load the model
model = AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype="auto",device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Call the model
# Define the prompt
prompt = "你好，请介绍下你自己。"
# Wrap the prompt into a message
message = [{"role":"system","content":"You are a helpful assistant system"},{"role":"user","content":prompt}]
# Use the tokenizer's apply_chat_template() method to convert the above-defined message list; tokenize=False means no tokenization at this point
text = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=True)

# Tokenize the processed text and convert it into the model's input tensor
model_inputs = tokenizer([text],return_tensors="pt").to(DEVICE)
# Input the model to get the output
generated = model.generate(model_inputs.input_ids,max_new_tokens=512)
print(generated)

# Decode and restore the output content
responce = tokenizer.batch_decode(generated,skip_special_tokens=True)
print(responce)
