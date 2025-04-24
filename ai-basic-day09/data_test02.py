import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Read the CSV file
csv_file_path = "/Users/shokhzodjuraev/Desktop/AI-Duke/AI-BasicStart/AI-Duke/ai-basic-day09/data/Weibo/validation.csv"
df = pd.read_csv(csv_file_path)

# Define the resampling strategy
# If you want oversampling, use RandomOverSampler
# If you want undersampling, use RandomUnderSampler
# Here we use RandomUnderSampler for undersampling
# random_state controls the seed for the random number generator
rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)

# Separate features and labels
X = df[["text"]]
Y = df[["label"]]
print(Y)
# Apply resampling
X_resampled, Y_resampled = rus.fit_resample(X, Y)
print(Y_resampled)
# Combine features and labels to create a new DataFrame
df_resampled = pd.concat([X_resampled, Y_resampled], axis=1)

print(df_resampled)

# Save the balanced data to a new CSV file
df_resampled.to_csv("validation.csv", index=False)