# Customized content generation
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Mr.Zhang\.cache\huggingface\hub\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
model = AutoModelForCausalLM.from_pretrained(r"C:\Users\Mr.Zhang\.cache\huggingface\hub\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
# Load our own trained weights (Chinese ancient poetry)
model.load_state_dict(torch.load("net.pt"))


# Define a function to generate 5-character quatrains. `text` is the prompt, `row` is the number of lines to generate, and `col` is the number of characters per line.
def generate(text, row, col):

    # Define an internal recursive function for text generation
    def generate_loop(data):
        # Disable gradient computation
        with torch.no_grad():
            # Use the data dictionary as model input and get the output
            out = model(**data)
        # Get the last character (logits are unnormalized probability outputs)
        out = out["logits"]
        # Select the last logits of each sequence, corresponding to the prediction of the next word
        out = out[:, -1]

        # Find the top 50 values by probability, and use this as a threshold to discard smaller values
        topk_value = torch.topk(out, 50).values
        # Get the top 50 largest logits for each output sequence (to maintain the original dimensions, add a dimension because indexing reduces dimensions)
        topk_value = topk_value[:, -1].unsqueeze(dim=1)
        # Set logits smaller than the 50th largest value to negative infinity to reduce the selection of low-probability words
        out = out.masked_fill(out < topk_value, -float("inf"))

        # Set the logits of special symbols to negative infinity to prevent the model from generating these symbols
        for i in ",.()《《[]「」{}":
            out[:, tokenizer.get_vocab()[i]] = -float('inf')

        # Perform sampling based on probabilities without replacement to avoid generating duplicate content
        out = out.softmax(dim=1)
        # Sample from the probability distribution to select the next word's ID
        out = out.multinomial(num_samples=1)

        # Forcefully add punctuation
        # Calculate the ratio of the current generated text length to the expected length
        c = data["input_ids"].shape[1] / (col + 1)
        # If the current length is an integer multiple of the expected length, add punctuation
        if c % 1 == 0:
            if c % 2 == 0:
                # Add a period at even positions
                out[:, 0] = tokenizer.get_vocab()["."]
            else:
                # Add a comma at odd positions
                out[:, 0] = tokenizer.get_vocab()[","]
        # Append the newly generated word ID to the end of the input sequence
        data["input_ids"] = torch.cat([data["input_ids"], out], dim=1)
        # Update the attention mask to mark all valid positions
        data["attention_mask"] = torch.ones_like(data["input_ids"])
        # Update token type IDs, usually used in BERT models, but not in GPT models
        data["token_type_ids"] = torch.ones_like(data["input_ids"])
        # Update labels by copying input IDs, typically used in language generation models to predict the next word
        data["labels"] = data["input_ids"].clone()

        # Check if the generated text length has reached or exceeded the specified number of rows and columns
        if data["input_ids"].shape[1] >= row * col + row + 1:
            # If the length requirement is met, return the final data dictionary
            return data
        # If the length requirement is not met, recursively call the `generate_loop` function to continue generating text
        return generate_loop(data)

    # Generate 3 poems
    # Use the tokenizer to encode the input text and repeat it 3 times to generate 3 samples
    data = tokenizer.batch_encode_plus([text] * 3, return_tensors="pt")
    # Remove the last token (end symbol) from the encoded sequence
    data["input_ids"] = data["input_ids"][:, :-1]
    # Create an all-ones tensor with the same shape as `input_ids` for the attention mask
    data["attention_mask"] = torch.ones_like(data["input_ids"])
    # Create an all-zeros tensor with the same shape as `input_ids` for token type IDs
    data["token_type_ids"] = torch.zeros_like(data["input_ids"])
    # Copy `input_ids` to `labels` for the model's target
    data['labels'] = data["input_ids"].clone()

    # Call the `generate_loop` function to start generating text
    data = generate_loop(data)

    # Iterate through the 3 generated samples
    for i in range(3):
        # Print the sample index and the corresponding decoded text
        print(i, tokenizer.decode(data["input_ids"][i]))

if __name__ == '__main__':
    generate("白", row=4, col=5)

