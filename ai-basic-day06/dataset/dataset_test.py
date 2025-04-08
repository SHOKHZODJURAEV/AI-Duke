from datasets import load_dataset, load_from_disk

# Load data online
# dataset = load_dataset(path="NousResearch/hermes-function-calling-v1",split="train")
# print(dataset)
# Load data from local disk
dataset = load_from_disk(r"/Users/shokhzodjuraev/Desktop/AI-Duke/AI-BasicStart/AI-Duke/ai-basic-day06/data/ChnSentiCorp")
print(dataset)

# Extract the test set
test_data = dataset["test"]
print(test_data)
# View the data
for data in test_data:
    print(data)