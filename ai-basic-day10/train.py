from transformers import AdamW
from transformers.optimization import get_scheduler
import torch
from data import MyDataset  # Import the custom dataset class
from transformers import AutoModelForCausalLM, AutoTokenizer  # Import the model and tokenizer classes from transformers
from torch.utils.data import DataLoader  # Import PyTorch's DataLoader class

# Instantiate the custom dataset
dataset = MyDataset()  # Create the dataset object

# Load the pre-trained tokenizer for text encoding
tokenizer = AutoTokenizer.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
# Load the pre-trained model for language modeling tasks
model = AutoModelForCausalLM.from_pretrained(r"E:\llm\gpt2-chinese\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")

# Define a function to convert text data into the format required by the model
def collate_fn(data):
    # Use the tokenizer to encode the data, and pad or truncate to a fixed length
    data = tokenizer.batch_encode_plus(data,
                                       padding=True,  # Pad sequences
                                       truncation=True,  # Truncate sequences
                                       max_length=512,  # Maximum length
                                       return_tensors='pt')  # Return PyTorch tensors
    # Copy input IDs as labels for language model training
    data['labels'] = data['input_ids'].clone()
    return data

# Use DataLoader to create a data loader for batch loading
loader = DataLoader(
    dataset=dataset,  # Specify the dataset
    batch_size=2,  # Specify the batch size
    shuffle=True,  # Shuffle the data
    drop_last=True,  # Drop the last batch if its size is less than batch_size
    collate_fn=collate_fn  # Specify how to collate samples into batches
)
print(f"Data length: {len(loader)}")  # Print the number of batches in the data loader

# Define the training function
def train():
    # Define training parameters
    EPOCH = 3000  # Number of training epochs
    global model  # Use the global model variable
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Check if GPU is available, use it if so, otherwise use CPU
    model = model.to(DEVICE)  # Move the model to the specified device

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)  # Use the AdamW optimizer and set the learning rate
    # Define the learning rate scheduler
    scheduler = get_scheduler(name="linear",  # Linear scheduler
                              num_warmup_steps=0,  # Number of warmup steps
                              num_training_steps=len(loader),  # Total number of training steps
                              optimizer=optimizer)
    model.train()  # Set the model to training mode
    for epoch in range(EPOCH):  # Loop through each training epoch
        for i, data in enumerate(loader):  # Iterate over batches in the data loader
            for k in data.keys():  # Move data to the specified device
                data[k] = data[k].to(DEVICE)
            out = model(**data)  # Forward pass
            loss = out['loss']  # Get the loss

            loss.backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to prevent explosion
            optimizer.step()  # Update model parameters
            scheduler.step()  # Update the learning rate

            optimizer.zero_grad()  # Clear the optimizer's gradients
            model.zero_grad()  # Clear the model's gradients

            if i % 50 == 0:  # Print information every 50 batches
                labels = data["labels"][:, 1:]  # Get the true labels, ignoring the <bos> token
                out = out["logits"].argmax(dim=2)[:,:-1]  # Get the predicted results, ignoring the <eos> token

                select = labels != 0  # Select non-padding labels
                labels = labels[select]  # Apply selection
                out = out[select]  # Apply selection
                del select  # Delete the no longer used select
                # Calculate accuracy
                acc = (labels == out).sum().item() / labels.numel()  # Formula to calculate accuracy
                lr = optimizer.state_dict()["param_groups"][0]['lr']  # Get the current learning rate

                # Print training information
                print(f"epoch:{epoch},batch:{i},loss:{loss.item()},lr:{lr},acc:{acc}")

        # Save the model parameters after the last epoch
        torch.save(model.state_dict(), "params/net.pt")  # Save the model parameters to the specified path
        print("Weights saved successfully!")  # Print success message

# When this script is run as the main program, call the training function
if __name__ == '__main__':
    train()  # Start the training process

# do_sample =True,  # Enable sampling look at high level usage of the word
# do_sample-False,  # Disable sampling look at from high to less level usage of the word


