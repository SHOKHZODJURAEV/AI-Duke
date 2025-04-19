from datasets import load_dataset,load_from_disk

# Load data online
# dataset = load_dataset(path="NousResearch/hermes-function-calling-v1",split="train")
# print(dataset)
# Save as CSV format
# dataset.to_csv(path_or_buf=r"D:\PycharmProjects\disanqi\demo_5\data\hermes-function-calling-v1.csv")
# Load data in CSV format
# dataset = load_dataset(path="csv",data_files=r"D:\PycharmProjects\disanqi\demo_5\data\hermes-function-calling-v1.csv")
# print(dataset)
# Load cached data
dataset = load_from_disk(r"D:\PycharmProjects\disanqi\demo_5\data\ChnSentiCorp")
print(dataset)
#
# test_data = dataset["train"]
# for data in test_data:
#     print(data)