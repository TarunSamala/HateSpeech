
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam
import pickle
import numpy as np

# Load preprocessed data
X_train = np.load('preprocessed/X_train.npy')
X_test = np.load('preprocessed/X_test.npy')
y_train_task1 = np.load('preprocessed/y_train_task1.npy')
y_test_task1 = np.load('preprocessed/y_test_task1.npy')
y_train_task2 = np.load('preprocessed/y_train_task2.npy')
y_test_task2 = np.load('preprocessed/y_test_task2.npy')

# Model Parameters
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE = 0.3

# Load the tokenizer
with open('preprocessed/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.word_index) + 1

# Build the Bi-LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
    Dropout(DROPOUT_RATE),
    Bidirectional(LSTM(LSTM_UNITS)),
    Dropout(DROPOUT_RATE),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Adjust for binary classification in task_1 or task_2
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_task1, epochs=10, batch_size=32, validation_data=(X_test, y_test_task1))

# Save the model
model.save('model/bilstm_model_task1.h5')

# Evaluate the model on test set for Task 1
test_loss, test_accuracy = model.evaluate(X_test, y_test_task1)
print(f"Test Loss for Task 1: {test_loss:.4f}")
print(f"Test Accuracy for Task 1: {test_accuracy:.4f}")
