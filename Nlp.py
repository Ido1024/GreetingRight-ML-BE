import pandas as pd
import re
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random
import tensorflow as tf
import os

# === Setup ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()


# === Preprocessing ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in STOPWORDS])


def process_text(text):
    tokens = tokenizer.tokenize(text)
    return " ".join([lemmatizer.lemmatize(token) for token in tokens])


def preprocess_input(text):
    return process_text(remove_stopwords(clean_text(text)))


# === Load user request training data ===
filepath = 'birthday_wish_requests.csv'
df = pd.read_csv(filepath)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df['user_request'] = df['user_request'].astype(str).fillna('').str.strip()
df['user_request'] = df['user_request'].str.replace(r'^"|"$', '', regex=True)
df = df[df['user_request'].str.split().str.len() > 3]
df['lemmatized_text'] = df['user_request'].apply(preprocess_input)

tone_cols = ['Happy', 'Funny', 'Rhyming_Poem', 'Heartfelt', 'Short_and_Simple', 'Inspirational', 'Professional']
rel_cols = ['Father', 'Mother', 'Wife', 'Husband', 'Boyfriend', 'Girlfriend', 'Son', 'Daughter',
            'Grandfather', 'Grandmother', 'Friend', 'Boss', 'Best-Friend']

df = df[(df[tone_cols + rel_cols].sum(axis=1) > 0)]
df.drop_duplicates(subset='lemmatized_text', inplace=True)

X = df['lemmatized_text']
y_tone = df[tone_cols].values
rel_encoder = LabelEncoder()
y_rel = rel_encoder.fit_transform(df[rel_cols].idxmax(axis=1))

X_train, X_test, y_tone_train, y_tone_test, y_rel_train, y_rel_test = train_test_split(
    X, y_tone, y_rel, test_size=0.2, stratify=y_rel, random_state=SEED
)

# === Load and preprocess wish texts ===
wishes_df = pd.read_csv('big_dataset.csv')
wishes_df['quote'] = wishes_df['quote'].astype(str).fillna('')
wishes_df['lemmatized_quote'] = wishes_df['quote'].apply(preprocess_input)

vectorizer = TfidfVectorizer(max_features=7000, stop_words='english', min_df=1)
vectorizer.fit(wishes_df['lemmatized_quote'])

X_train_vec = vectorizer.transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

y_rel_train_cat = to_categorical(y_rel_train)
y_rel_test_cat = to_categorical(y_rel_test)

# === Model: Tone ===
model_tone = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_vec.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(len(tone_cols), activation='sigmoid')
])
model_tone.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Model: Relationship ===
model_rel = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_vec.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(len(rel_cols), activation='softmax')
])
model_rel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Training ===
callbacks = [
    EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
    ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.3)
]

model_tone.fit(X_train_vec, y_tone_train, validation_split=0.1, epochs=20, batch_size=32, callbacks=callbacks)
model_rel.fit(X_train_vec, y_rel_train_cat, validation_split=0.1, epochs=20, batch_size=32, callbacks=callbacks)

# === Evaluation ===

# Relationship Evaluation (Multi-class)
y_rel_pred_probs = model_rel.predict(X_test_vec)
y_rel_pred = np.argmax(y_rel_pred_probs, axis=1)

print("\n=== Accuracy per Relationship ===")
for i, label in enumerate(rel_encoder.classes_):
    mask = (y_rel_test == i)
    acc = accuracy_score(y_rel_test[mask], y_rel_pred[mask]) if np.any(mask) else 0
    print(f"{label}: {acc:.2f}")

# Tone Evaluation (Multi-label)
y_tone_pred_probs = model_tone.predict(X_test_vec)
y_tone_pred = (y_tone_pred_probs > 0.5).astype(int)

print("\n=== Accuracy per Tone ===")
for i, tone in enumerate(tone_cols):
    y_true_col = y_tone_test[:, i]
    y_pred_col = y_tone_pred[:, i]
    acc = accuracy_score(y_true_col, y_pred_col)
    print(f"{tone}: {acc:.2f}")


# === Inference Functions ===
def predict_tones(text):
    prepped = preprocess_input(text)
    tfidf = vectorizer.transform([prepped])
    probs = model_tone.predict(tfidf.toarray())[0]
    return [tone_cols[i] for i, p in enumerate(probs) if p > 0.5] or ["Happy"]


def predict_rel(text):
    prepped = preprocess_input(text)
    tfidf = vectorizer.transform([prepped])
    probs = model_rel.predict(tfidf.toarray())[0]
    return rel_encoder.inverse_transform([np.argmax(probs)])[0]


# Expose variables
__all__ = [
    'vectorizer', 'model_tone', 'model_rel',
    'rel_encoder', 'tone_cols', 'rel_cols',
    'predict_tones', 'predict_rel', 'preprocess_input'
]
