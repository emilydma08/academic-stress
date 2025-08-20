import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Importing CSV file
df = pd.read_csv('./data/student_lifestyle_dataset.csv')


#Data Preprocessing
columns_to_standardize = ['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'GPA']
df['Stress_Level'] = df["Stress_Level"].map({"Low": 0, "Moderate": 1, "High":2}).astype(float)

scaler = StandardScaler()
df[columns_to_standardize] = df[columns_to_standardize].fillna(df[columns_to_standardize].mean())
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

X = df.drop(columns=['Stress_Level', 'Student_ID']).to_numpy()
y = df['Stress_Level'].to_numpy()

#Saving the scaler
#joblib.dump(scaler, "scaler.pkl")


#Data Transformer Classes
class CustomTransform():
    def __call__(self, sample):
        X, y = sample
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return X, y

class CustomDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        num_rows_in_dataset = len(self.X)
        return num_rows_in_dataset

    def __getitem__(self, idx):
        sample = (self.X[idx], self.y[idx])

        if self.transform:
          sample = self.transform(sample)

        return sample

transform = CustomTransform()
dataset = CustomDataset(X, y, transform)

total_size = len(dataset)
train_size = int(0.7 * total_size)
valid_size = int(0.15 * total_size)
test_size = total_size - train_size - valid_size

train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

