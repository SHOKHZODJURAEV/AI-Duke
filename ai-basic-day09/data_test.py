import pandas as pd

df = pd.read_csv("data/news/train.csv")
# Count the number of data points for each category
category_counts = df["label"].value_counts()

# Calculate the ratio of each category
total_data = len(df)
category_ratios = (category_counts / total_data) *100

print(category_counts)
print(category_ratios)