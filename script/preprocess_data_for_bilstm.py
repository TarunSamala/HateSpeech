
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load dataset
data = pd.read_csv('data/en_Hasoc2021_train.csv')

# Preprocess text data
def clean_text(text):
    # Remove URLs, special characters, and punctuation
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()

data['text'] = data['text'].apply(clean_text)

# Tokenize and pad text sequences
MAX_VOCAB_SIZE = 10000  # Limit on vocabulary size
MAX_SEQUENCE_LENGTH = 100  # Limit each text to 100 tokens

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Encode labels
label_encoder_task1 = LabelEncoder()
label_encoder_task2 = LabelEncoder()
data['task_1_encoded'] = label_encoder_task1.fit_transform(data['task_1'])
data['task_2_encoded'] = label_encoder_task2.fit_transform(data['task_2'])

# Train-test split
X_train, X_test, y_train_task1, y_test_task1, y_train_task2, y_test_task2 = train_test_split(
    padded_sequences, 
    data['task_1_encoded'], 
    data['task_2_encoded'], 
    test_size=0.2, 
    random_state=42
)
# Save the preprocessed data for model training
np.save('preprocessed/X_train.npy', X_train)
np.save('preprocessed/X_test.npy', X_test)
np.save('preprocessed/y_train_task1.npy', y_train_task1)
np.save('preprocessed/y_test_task1.npy', y_test_task1)
np.save('preprocessed/y_train_task2.npy', y_train_task2)
np.save('preprocessed/y_test_task2.npy', y_test_task2)


# Save tokenizer and label encoders for later use
import pickle
with open('preprocessed/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('preprocessed/label_encoder_task1.pkl', 'wb') as f:
    pickle.dump(label_encoder_task1, f)
with open('preprocessed/label_encoder_task2.pkl', 'wb') as f:
    pickle.dump(label_encoder_task2, f)

print("Data preprocessed and saved for Bi-LSTM model training.")
