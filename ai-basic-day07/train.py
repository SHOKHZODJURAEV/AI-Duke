# Model training
import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer, AdamW

# Define device information
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the number of training epochs
EPOCH = 30000

token = BertTokenizer.from_pretrained(r"/Users/shokhzodjuraev/Desktop/AI-Duke/AI-BasicStart/AI-Duke/ai-basic-day07/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

def collate_fn(data):
    sents = [i[0] for i in data]
    label = [i[1] for i in data]
    # Encoding
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=500,
        padding="max_length",
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)

    return input_ids, attention_mask, token_type_ids, labels


# Create dataset
train_dataset = MyDataset("train")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=100,
    shuffle=True,
    # Drop the last batch of data to prevent shape mismatch
    drop_last=True,
    # Encode the loaded data
    collate_fn=collate_fn
)

if __name__ == '__main__':
    # Start training
    print(DEVICE)
    model = Model().to(DEVICE)
    # Define optimizer
    optimizer = AdamW(model.parameters())
    # Define loss function
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # Move data to DEVICE
            input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
            # Forward computation (input data into the model to get output)
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # Compute loss based on output
            loss = loss_func(out, labels)
            # Optimize parameters based on loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Output training information every 5 batches
            if i % 5 == 0:
                out = out.argmax(dim=1)
                acc = (out == labels).sum().item() / len(labels)
                print(f"epoch:{epoch},i:{i},loss:{loss.item()},acc:{acc}")
        # Save parameters after each epoch
        torch.save(model.state_dict(), f"params/{epoch}_bert.pth")
        print(epoch, "Parameters saved successfully!")