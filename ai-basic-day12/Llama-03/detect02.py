# Customized content generation
from transformers import AutoModelForCausalLM,AutoTokenizer,TextGenerationPipeline
import torch

tokenizer = AutoTokenizer.from_pretrained(r"D:\PycharmProjects\day_18demo\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
model = AutoModelForCausalLM.from_pretrained(r"D:\PycharmProjects\day_18demo\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")

# Load our custom-trained weights (Chinese poetry)
model.load_state_dict(torch.load("net.pt"))

# Define a function to generate 5-character quatrains. 'text' is the prompt, 'row' is the number of rows, and 'col' is the number of characters per row.
def generate(text,row,col):
    # Define an internal recursive function for text generation
    def generate_loop(data):
        # Disable gradient computation
        with torch.no_grad():
            # Use the dictionary data as input to the model and get the output
            out = model(**data)
        # Get the last logits (unnormalized probability outputs)
        out = out["logits"]
        # Select the last logits of each sequence, corresponding to the prediction of the next word
        out = out[:,-1]

        # Find the top 50 values by probability, and use this as a threshold, discarding all values below it
        topk_value = torch.topk(out,50).values
        # Get the top 50 largest logits for each output sequence (to maintain the original dimension, we need to add a dimension to the result, because the indexing operation reduces the dimension)
        topk_value = topk_value[:,-1].unsqueeze(dim=1)
        # Set the logits of all values smaller than the 50th largest to negative infinity, reducing the likelihood of selecting low-probability words
        out = out.masked_fill(out< topk_value,-float("inf"))
        # Mask [UNK] token
        out[:, tokenizer.get_vocab()["[UNK]"]] = -float("inf")
        # Set logits of special symbols to negative infinity to prevent the model from generating these symbols。
        for i in ",.()《》[]「」{}":
            out[:,tokenizer.get_vocab()[i]] = -float("inf")

        # Perform sampling based on probabilities without replacement to avoid generating duplicate content
        out = out.softmax(dim=1)
        # Randomly sample from the probability distribution to select the next word ID
        out = out.multinomial(num_samples=1)

        # Force punctuation
        # Calculate the ratio of the current generated text length to the expected length
        c = data["input_ids"].shape[1] / (col+1)

        # If the current length is an integer multiple of the expected length, add punctuation
        if c %1 ==0:
            if c%2==0:
                # Add a period at even positions
                out[:,0] = tokenizer.get_vocab()["."]
            else:
                # Add a comma at odd positions
                out[:,0] = tokenizer.get_vocab()[","]

        # Append the generated new word ID to the end of the input sequence
        data["input_ids"] = torch.cat([data["input_ids"],out],dim=1)
        # Update the attention mask to mark all valid positions
        data["attention_mask"] = torch.ones_like(data["input_ids"])
        # Update token type IDs, typically used in BERT models but not in GPT
        data["token_type_ids"] = torch.ones_like(data["input_ids"])
        # Update labels by copying input IDs, usually used in language generation models to predict the next word
        data["labels"] = data["input_ids"].clone()

        # Check if the generated text length has reached or exceeded the specified rows and columns
        if data["input_ids"].shape[1] >= row*col + row+1:
            # If the length requirement is met, return the final data dictionary
            return data
        # If the length requirement is not met, recursively call the generate_loop function to continue generating text
        return generate_loop(data)

    # Generate 3 poems
    # Use the tokenizer to encode the input text and repeat it 3 times to generate 3 samples
    data = tokenizer.batch_encode_plus([text]*3,return_tensors="pt")
    # Remove the last token (end symbol) from the encoded sequence
    data["input_ids"] = data["input_ids"][:,:-1]
    # Create an all-ones tensor with the same shape as input_ids for the attention mask
    data["attention_mask"] = torch.ones_like(data["input_ids"])
    # Create an all-ones tensor with the same shape as input_ids for token type IDs
    data["token_type_ids"] = torch.ones_like(data["input_ids"])
    # Copy input_ids to labels for model targets
    data["labels"] = data["input_ids"].clone()

    # Call the generate_loop function to start generating text
    data = generate_loop(data)

    # Iterate through the 3 generated samples
    for i in range(3):
        print(i,tokenizer.decode(data["input_ids"][i]))

if __name__ == '__main__':
    generate("白",row=4,col=5)