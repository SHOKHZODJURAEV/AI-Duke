from transformers import AdamW
import torch
from data import MyDataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from tensorboardX import SummaryWriter

# Load dataset
dataset = MyDataset()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('/root/app/huggingface/LLM/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3/')
# Load model
model = AutoModelForCausalLM.from_pretrained(
    '/root/app/huggingface/LLM/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3/')

# Data preprocessing function
def collate_fn(data):
    data = tokenizer.batch_encode_plus(data,
                                       padding=True,
                                       truncation=True,
                                       max_length=512,
                                       return_tensors='pt')

    data['labels'] = data['input_ids'].clone()

    return data

# Data loader
loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=13,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
)

# Create an instance of TensorBoard's SummaryWriter
writer = SummaryWriter("/root/app/projects/day17_demo/logdir/")

# Training function
def train():
    global model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    torch.cuda.empty_cache()  # Clear cache memory before training starts
    for epoch in range(1000):
        for i, data in enumerate(loader):
            for k in data.keys():
                data[k] = data[k].to(device)
            out = model(**data)
            loss = out['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                with torch.no_grad():
                    labels = data['labels'][:, 1:]
                    out = out['logits'].argmax(dim=2)[:, :-1]
                    select = labels != 0
                    labels = labels[select]
                    out = out[select]
                    accuracy = (labels == out).sum().item() / labels.numel()
                    del labels, out  # 删除变量以释放内存

                lr = optimizer.state_dict()['param_groups'][0]['lr']

                print(f"Epoch: {epoch}, Step: {i}, Loss: {loss.item()}, LR: {lr}, Accuracy: {accuracy}")

                # Write metrics to TensorBoard
                writer.add_scalar('Loss/train', loss.item(), epoch * len(loader) + i)
                writer.add_scalar('Accuracy/train', accuracy, epoch * len(loader) + i)
                writer.add_scalar('Learning Rate', lr, epoch * len(loader) + i)

        # Save model parameters, without saving the model structure
        torch.save(model.state_dict(), 'net.pt')
        print(f"Epoch {epoch} - Weights saved successfully!")

        torch.cuda.empty_cache()  # Clear unused cache memory

# Main program entry point
if __name__ == '__main__':
    train()

# Close SummaryWriter
writer.close()