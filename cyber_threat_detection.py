import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN, Bidirectional, Conv1D, GlobalMaxPooling1D

# =========================
# LOAD DATA
# =========================
def load_data(file_path):
    df = pd.read_csv(file_path).dropna(subset=['Payload Data', 'Malware Indicators'])
    le = LabelEncoder()
    df['Malware Label'] = le.fit_transform(df['Malware Indicators'])
    return train_test_split(df['Payload Data'], df['Malware Label'], test_size=0.2, random_state=42)

# =========================
# DATASET CLASS (RoBERTa)
# =========================
class CyberDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, return_tensors='pt')
        self.labels = torch.tensor(labels.values)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# =========================
# RoBERTa MODEL
# =========================
def train_roberta(X_train, X_test, y_train, y_test):
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)

    train_dataset = CyberDataset(X_train, y_train, tokenizer)
    test_dataset = CyberDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).train()

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss = model(input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        optimizer.step()

    model.eval()
    predictions = []
    test_loader = DataLoader(test_dataset, batch_size=8)

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())

    return accuracy_score(y_test, predictions)

# =========================
# COMMON TOKENIZER (for DL models)
# =========================
def prepare_sequences(X_train, X_test):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
    X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

    return X_train_pad, X_test_pad

# =========================
# LSTM
# =========================
def train_lstm(X_train_pad, X_test_pad, y_train, y_test):
    model = Sequential([
        Embedding(5000, 128, input_length=100),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=2, batch_size=32, verbose=1)

    preds = (model.predict(X_test_pad) > 0.5).astype("int32")
    return accuracy_score(y_test, preds)

# =========================
# BiRNN
# =========================
def train_birnn(X_train_pad, X_test_pad, y_train, y_test):
    model = Sequential([
        Embedding(5000, 128, input_length=100),
        Bidirectional(SimpleRNN(64)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=2, batch_size=32, verbose=1)

    preds = (model.predict(X_test_pad) > 0.5).astype("int32")
    return accuracy_score(y_test, preds)

# =========================
# CNN
# =========================
def train_cnn(X_train_pad, X_test_pad, y_train, y_test):
    model = Sequential([
        Embedding(5000, 128, input_length=100),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=2, batch_size=32, verbose=1)

    preds = (model.predict(X_test_pad) > 0.5).astype("int32")
    return accuracy_score(y_test, preds)

# =========================
# DNN
# =========================
def train_dnn(X_train_pad, X_test_pad, y_train, y_test):
    model = Sequential([
        Embedding(5000, 128, input_length=100),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=2, batch_size=32, verbose=1)

    preds = (model.predict(X_test_pad) > 0.5).astype("int32")
    return accuracy_score(y_test, preds)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("cyb.csv")

    # small dataset (as you used)
    X_train, X_test = X_train[:500], X_test[:100]
    y_train, y_test = y_train[:500], y_test[:100]

    # RoBERTa
    roberta_acc = train_roberta(X_train, X_test, y_train, y_test)

    # Other models
    X_train_pad, X_test_pad = prepare_sequences(X_train, X_test)

    lstm_acc = train_lstm(X_train_pad, X_test_pad, y_train, y_test)
    birnn_acc = train_birnn(X_train_pad, X_test_pad, y_train, y_test)
    cnn_acc = train_cnn(X_train_pad, X_test_pad, y_train, y_test)
    dnn_acc = train_dnn(X_train_pad, X_test_pad, y_train, y_test)

    print("\n===== FINAL RESULTS =====")
    print(f"RoBERTa Accuracy: {roberta_acc}")
    print(f"LSTM Accuracy:    {lstm_acc}")
    print(f"BiRNN Accuracy:   {birnn_acc}")
    print(f"CNN Accuracy:     {cnn_acc}")
    print(f"DNN Accuracy:     {dnn_acc}")
