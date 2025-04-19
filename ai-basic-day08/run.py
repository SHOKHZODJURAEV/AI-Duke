import torch
from net import Model
from transformers import BertTokenizer

# Define device information
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

token = BertTokenizer.from_pretrained(r"D:\PycharmProjects\disanqi\demo_5\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
names = ["Negative Review", "Positive Review"]
model = Model().to(DEVICE)

def collate_fn(data):
    sents = []
    sents.append(data)
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

    return input_ids, attention_mask, token_type_ids

def test():
    # Load training parameters
    model.load_state_dict(torch.load("params/1_bert.pth", map_location=DEVICE))
    # Enable test mode
    model.eval()

    while True:
        data = input("Please enter test data (enter 'q' to quit): ")
        if data == 'q':
            print("Testing ended")
            break
        input_ids, attention_mask, token_type_ids = collate_fn(data)
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE), attention_mask.to(DEVICE), \
            token_type_ids.to(DEVICE)

        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
            out = out.argmax(dim=1)
            print("Model judgment: ", names[out], "\n")

if __name__ == '__main__':
    test()