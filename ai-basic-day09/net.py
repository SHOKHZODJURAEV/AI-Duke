from transformers import BertModel,BertConfig
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
# pretrained = BertModel.from_pretrained(r"E:\PycharmProjects\demo_7\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f").to(DEVICE)
# pretrained.embeddings.position_embeddings = torch.nn.Embedding(1024,768).to(DEVICE)
config = BertConfig.from_pretrained(r"E:\PycharmProjects\demo_7\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
config.max_position_embeddings = 1024
print(config)

# Initialize the model using the configuration file
pretrained = BertModel(config).to(DEVICE)
print(pretrained)
# Define downstream tasks
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768,10)
    def forward(self,input_ids,attention_mask,token_type_ids):
        # Freeze the pre-trained model weights
        # with torch.no_grad():
        # out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Incremental model participates in training
        out = self.fc(out.last_hidden_state[:,0])
        return out
