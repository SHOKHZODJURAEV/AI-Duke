from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set the directory containing config.json
model_dir = r"/trasnFormers_test/model/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Use the loaded model and tokenizer to create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")

# Generate text
# output = generator("Hello, I am a language model,", max_length=50, num_return_sequences=1)
# output = generator("Hello, I am a language model,", max_length=50, num_return_sequences=1, truncation=True, clean_up_tokenization_spaces=False)
output = generator(
    "Hello, I am a language model,",  # Input seed text (prompt) for text generation. The model will generate subsequent text based on this initial text.
    max_length=50,  # Specify the maximum length of the generated text. Here, 50 means the generated text contains at most 50 tokens.
    num_return_sequences=2,  # Specify how many independent text sequences to return. A value of 1 means only one text sequence is generated and returned.
    truncation=True,  # Determines whether to truncate the input text to fit the model's maximum input length. If True, the part exceeding the maximum input length will be truncated; if False, the model may fail to process overly long input and throw an error.
    temperature=0.7,  # Controls the randomness of the generated text. Lower values make the text more conservative (favoring higher-probability words), while higher values make the text more diverse (favoring a wider range of words). 0.7 is a common setting that balances randomness and coherence.
    top_k=50,  # Limits the model to consider only the top k words with the highest probabilities at each step of generation. Here, top_k=50 means the model considers only the top 50 candidate words for each step, reducing the likelihood of generating improbable words.
    top_p=0.9,  # Also known as nucleus sampling, further restricts the range of words the model can choose from during generation. It selects a set of words whose cumulative probability reaches p. top_p=0.9 means the model will sample the next word from the most probable 90% of words, improving the quality of the generated text.
    clean_up_tokenization_spaces=True  # Controls whether to clean up spaces introduced during tokenization in the generated text. If set to True, extra spaces are removed; if False, they are retained. The default value is changing to False, as it better preserves the original text format.
)
print(output)
