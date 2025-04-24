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

token = BertTokenizer.from_pretrained(r"E:\PycharmProjects\demo_7\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

def collate_fn(data):
    sents = [i[0]for i in data]
    label = [i[1] for i in data]
    # Encode
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)

    return input_ids,attention_mask,token_type_ids,labels


# Create dataset
train_dataset = MyDataset("train")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=True,
    # Discard the last batch of data to prevent shape errors
    drop_last=True,
    # Encode the loaded data
    collate_fn=collate_fn
)

val_dataset = MyDataset("validation")
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=2,
    shuffle=True,
    # Discard the last batch of data to prevent shape errors
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

    # Initialize the best validation accuracy
    best_val_acc = 0.0

    for epoch in range(EPOCH):

        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # Store data on DEVICE
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
        # Validate the model (check for overfitting)
        # Set to evaluation mode
        model.eval()
        # No need for the model to participate in training
        with torch.no_grad():
            val_acc = 0.0
            val_loss = 0.0
            for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(val_loader):
                # Store data on DEVICE
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
                    DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
                # Forward computation (input data into the model to get output)
                out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                # Compute loss based on output
                val_loss += loss_func(out, labels)
                out = out.argmax(dim=1)
                val_acc += (out == labels).sum().item()
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            print(f"Validation set: loss:{val_loss},acc:{val_acc}")

            # Save the best parameters based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "params/best_bert.pth")
                print(f"Epoch:{epoch}: Saved best parameters: acc:{best_val_acc}")

        # Save the parameters of the last epoch
        torch.save(model.state_dict(), f"params/last_bert.pth")
        print(epoch, f"Epoch:{epoch} Last epoch parameters saved successfully!")