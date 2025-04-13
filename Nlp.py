import pandas as pd
import re
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
import tensorflow as tf
import os

# 🧪 Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# 📥 Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# 📄 Load data
filepath = 'birthday_wish_requests.csv'
if not os.path.exists(filepath):
    raise FileNotFoundError(f"File not found: {filepath}")

df = pd.read_csv(filepath)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df['user_request'] = df['user_request'].astype(str).str.strip().fillna('')
df['user_request'] = df['user_request'].str.replace(r'^"|"$', '', regex=True)
df = df[df['user_request'].str.split().str.len() > 3]

# 🧹 Text Cleaning Pipeline
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in STOPWORDS])

tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

def process_text(text):
    tokens = tokenizer.tokenize(text)
    return " ".join([lemmatizer.lemmatize(token) for token in tokens])

df['clean_text'] = df['user_request'].apply(clean_text)
df['clean_text'] = df['clean_text'].apply(remove_stopwords)
df['lemmatized_text'] = df['clean_text'].apply(process_text)

# 🎯 Define labels
tone_cols = ['Happy', 'Funny', 'Rhyming_Poem', 'Heartfelt', 'Short_and_Simple', 'Inspirational', 'Professional']
rel_cols = ['Father', 'Mother', 'Wife', 'Husband', 'Boyfriend', 'Girlfriend', 'Son', 'Daughter', 'Grandfather',
            'Grandmother', 'Friend', 'Boss', 'Best-Friend']

df = df[(df[tone_cols + rel_cols].sum(axis=1) > 0)]
df.drop_duplicates(subset='lemmatized_text', inplace=True)

# 🧾 Features and Targets
X = df['lemmatized_text']
y_tone = df[tone_cols].values
rel_encoder = LabelEncoder()
y_rel = rel_encoder.fit_transform(df[rel_cols].idxmax(axis=1))

# 🧪 Train/Test Split
X_train, X_test, y_tone_train, y_tone_test, y_rel_train, y_rel_test = train_test_split(
    X, y_tone, y_rel, test_size=0.2, stratify=y_rel, random_state=SEED
)

# 🔡 TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=7000, stop_words='english', min_df=3)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🧠 Encode relationship labels
y_rel_train_cat = to_categorical(y_rel_train)
y_rel_test_cat = to_categorical(y_rel_test)

# 🏗️ Model: Tone (Multi-label)
model_tone = Sequential([
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(X_train_tfidf.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(len(tone_cols), activation='sigmoid')
])
model_tone.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🏗️ Model: Relationship (Single-label, multiclass)
model_rel = Sequential([
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(X_train_tfidf.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.3),
    Dense(len(rel_cols), activation='softmax')
])
model_rel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🧪 Training Callbacks
callbacks = [
    EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
    ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.3)
]

# 🧠 Train Models
model_tone.fit(X_train_tfidf.toarray(), y_tone_train,
               validation_split=0.1, epochs=20, batch_size=32,
               callbacks=callbacks, verbose=1)

model_rel.fit(X_train_tfidf.toarray(), y_rel_train_cat,
              validation_split=0.1, epochs=20, batch_size=32,
              callbacks=callbacks, verbose=1)

# 🧪 Evaluate
y_tone_pred = model_tone.predict(X_test_tfidf.toarray())
y_rel_pred = model_rel.predict(X_test_tfidf.toarray())
y_tone_pred_binary = (y_tone_pred > 0.5).astype(int)
y_rel_pred_class = np.argmax(y_rel_pred, axis=1)

print("\n📊 Multi-label Tone Classification Report:")
print(classification_report(y_tone_test, y_tone_pred_binary, target_names=tone_cols))

print("\n📊 Relationship Classification Report:")
print(classification_report(y_rel_test, y_rel_pred_class, target_names=rel_encoder.classes_))

# 🔍 Preprocessing for inference
def preprocess_input(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = process_text(text)
    return text

# 🔮 Prediction Functions
def predict_tones(text):
    prepped = preprocess_input(text)
    tfidf = vectorizer.transform([prepped])
    tone_probs = model_tone.predict(tfidf.toarray())[0]
    detected_tones = [tone_cols[i] for i, p in enumerate(tone_probs) if p > 0.5]
    return detected_tones if detected_tones else ["Happy"]

def predict_rel(text):
    prepped = preprocess_input(text)
    tfidf = vectorizer.transform([prepped])
    rel_probs = model_rel.predict(tfidf.toarray())[0]
    max_rel_index = np.argmax(rel_probs)
    return rel_encoder.inverse_transform([max_rel_index])[0] if rel_probs[max_rel_index] > 0 else "Friend"

def predict_request(text):
    prepped = preprocess_input(text)
    tfidf = vectorizer.transform([prepped])
    detected_tones = predict_tones(text)
    detected_rel = predict_rel(text)
    print("\n🔎 User Request:", text)
    print("💬 Detected Relationship:", detected_rel)
    print("🎭 Detected Tone(s):", ', '.join(detected_tones))
    rel_probs = model_rel.predict(tfidf.toarray())[0]
    sorted_rel = sorted(zip(rel_encoder.classes_, rel_probs), key=lambda x: x[1], reverse=True)
    print("\n📊 Relationship Prediction Confidence:")
    for rel, score in sorted_rel[:3]:
        print(f"{rel:<15}: {score:.2f}")

# 🧪 Try some examples
sample_requests = [
    "Create me a birthday wish for my mother, make it funny and long",
    "I want birthday wish that is happy and short for my girlfriend",
    "Write a birthday wish for the man who raised me",
    "Something for the love of my life on her special day",
    "Say something kind to the lady who brought me into life",
    "Write a birthday wish for my wife that's happy and rhyming",
    "Something professional and short for my boss",
    "Say something to my son that's funny and inspirational"
]

for req in sample_requests:
    predict_request(req)
