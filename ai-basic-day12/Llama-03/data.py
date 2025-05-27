import torch


# Define dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self):
        with open('data/chinese_poems.txt',encoding='utf-8') as f:
            lines = f.readlines()
        lines = [i.strip() for i in lines]

        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        return self.lines[i]


if __name__ == '__main__':
    dataset = MyDataset()

    print(len(dataset), dataset[0])