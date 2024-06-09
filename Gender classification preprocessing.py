import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch
import joblib

# Define the NameDataset class
class NameDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        item = {key: val.squeeze() for key, val in inputs.items()}
        item['labels'] = torch.tensor(label)
        return item

# Load data
data = pd.read_csv(r'C:\Users\UGBOKE GEORGE\Downloads\NationalNames\N_gram-name-prediction\NationalNames.csv', encoding= 'latin1')
data.head()

# Encode labels
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(data['Name'], data['Gender'], test_size=0.2)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = NameDataset(train_texts, train_labels)
val_dataset = NameDataset(val_texts, val_labels)

# Save datasets
joblib.dump(train_dataset, 'train_dataset.pkl')
joblib.dump(val_dataset, 'val_dataset.pkl')
