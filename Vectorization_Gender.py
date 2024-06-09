import torch
import joblib
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Load preprocessed data
train_texts, val_texts, train_labels, val_labels = joblib.load('processed_data.pkl')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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


train_dataset = NameDataset(train_texts, train_labels)
val_dataset = NameDataset(val_texts, val_labels)

joblib.dump(train_dataset, 'train_dataset.pkl')
joblib.dump(val_dataset, 'val_dataset.pkl')

# Just a print statement to verify everything works
print(f"Number of training samples: {len(train_dataset)}, Number of validation samples: {len(val_dataset)}")
