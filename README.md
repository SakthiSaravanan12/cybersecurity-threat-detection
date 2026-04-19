# 🛡️ Cybersecurity Threat Detection on Social Media Using Deep Learning

A comparative deep learning research project that detects cybersecurity threats such as **phishing, hate speech, and spam** from social media text data — comparing 5 different models to find the best performer.

---

## 📌 Project Overview

This project compares the performance of **XLM-RoBERTa, LSTM, Bidirectional RNN, CNN, and DNN** models for classifying cybersecurity threats in social media payloads. Each model is trained and evaluated on the same dataset to provide a fair comparison across Accuracy, Recall, and F1-Score metrics.

---

## 🚀 Models Compared

| Model | Architecture | Framework |
|-------|-------------|-----------|
| XLM-RoBERTa | Transformer-based pretrained model | PyTorch + Hugging Face |
| LSTM | Long Short-Term Memory | TensorFlow / Keras |
| BiRNN | Bidirectional Simple RNN | TensorFlow / Keras |
| CNN | 1D Convolutional Neural Network | TensorFlow / Keras |
| DNN | Deep Neural Network | TensorFlow / Keras |

---

## ✨ Key Features

- Detects threats like **phishing, malware indicators, hate speech, and spam**
- Compares **5 deep learning architectures** on the same dataset
- Uses **XLM-RoBERTa** (multilingual transformer) for best-in-class NLP performance
- Evaluates models on **Accuracy, Recall, and F1-Score**
- Clean modular code — each model is a separate function

---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch, TensorFlow, Keras
- **NLP / Transformers:** Hugging Face Transformers, XLM-RoBERTa
- **Data Handling:** Pandas, Scikit-learn
- **Preprocessing:** Tokenizer, TF-IDF, LabelEncoder, pad_sequences

---

## 📁 Project Structure

```
cybersecurity-threat-detection/
│
├── cyber_threat_detection.py   # Main code with all 5 models
├── cyb.csv                     # Dataset (Payload Data + Malware Indicators)
└── README.md
```

---

## ⚙️ How to Run

**1. Install dependencies**
```bash
pip install torch transformers scikit-learn pandas tensorflow
```

**2. Place your dataset**

Make sure `cyb.csv` is in the same folder with columns:
- `Payload Data` — text input
- `Malware Indicators` — label column

**3. Run the project**
```bash
python cyber_threat_detection.py
```

**4. Output**
```
===== FINAL RESULTS =====
RoBERTa Accuracy: 0.89
LSTM Accuracy:    0.82
BiRNN Accuracy:   0.85
CNN Accuracy:     0.84
DNN Accuracy:     0.80
```

---

## 📊 Results Summary

- **XLM-RoBERTa** achieved the best overall accuracy
- **BiRNN** showed improved Recall for threat detection
- **CNN** delivered the best F1-Score balance

---

## 👤 Author

**D Sakthi Saravanan**
Final Year B.E Computer Science Engineering
Saveetha Institute of Medical and Technical Sciences

---

## 📄 License

This project is for academic and research purposes.
