import torch
import Preprocessing
from transformers import BertTokenizer

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