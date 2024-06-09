import joblib
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Define the NameDataset class again
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

# Ensure the tokenizer is defined
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load datasets
train_dataset = joblib.load('train_dataset.pkl')
val_dataset = joblib.load('val_dataset.pkl')

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()
