import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer

# Define device information
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

token = BertTokenizer.from_pretrained(r"D:\PycharmProjects\disanqi\demo_5\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

def collate_fn(data):
    sents = [i[0]for i in data]
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

    return input_ids,attention_mask,token_type_ids,labels

# Create dataset
test_dataset = MyDataset("test")
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=100,
    shuffle=True,
    # Discard the last batch of data to prevent shape errors
    drop_last=True,
    # Encode the loaded data
    collate_fn=collate_fn
)

if __name__ == '__main__':
    acc = 0.0
    total = 0

    # Start testing
    print(DEVICE)
    model = Model().to(DEVICE)
    # Load training parameters
    model.load_state_dict(torch.load("params/3_bert.pth"))
    # Enable test mode for the model
    model.eval()
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        # Store data on DEVICE
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
            DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)
        # Forward computation (input data into the model to get output)
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        acc += (out==labels).sum().item()
        print(i,(out==labels).sum().item())
        total+=len(labels)
    print(f"test_acc：{acc/total}")