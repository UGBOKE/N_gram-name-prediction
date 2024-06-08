import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



# Load data
data = pd.read_csv(r'C:\Users\UGBOKE GEORGE\Downloads\NationalNames\N_gram-name-prediction\NationalNames.csv', encoding= 'latin1')
data.head()
# Encode gender labels
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
label_mapping # F = 0 and M = 1

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(data['Name'], data['Gender'], test_size=0.2)