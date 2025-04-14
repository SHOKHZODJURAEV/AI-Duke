from transformers import BertModel
import torch

# Define device information
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Load pre-trained model
pretrained = BertModel.from_pretrained(r"/Users/shokhzodjuraev/Desktop/AI-Duke/AI-BasicStart/AI-Duke/ai-basic-day07/model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f").to(DEVICE)
print(pretrained)
# Define downstream task (incremental model)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Design a fully connected network to implement a binary classification task
        self.fc = torch.nn.Linear(768,2)

    def forward(self,input_ids,attention_mask,token_type_ids):
        # Freeze the parameters of the Bert model so that it does not participate in training
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        # The incremental model participates in training
        out = self.fc(out.last_hidden_state[:,0])
        return out
