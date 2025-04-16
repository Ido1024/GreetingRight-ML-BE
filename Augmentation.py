import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Nlp import preprocess_input

class BirthdayWishAugmentor:
    def __init__(self, wishes_path, vectorizer, model_tone, model_rel, rel_encoder,
                 tone_cols, rel_cols, predict_tones_fn, predict_rel_fn):
        self.wishes_df = pd.read_csv(wishes_path).fillna('')
        self.vectorizer = vectorizer
        self.model_tone = model_tone
        self.model_rel = model_rel
        self.rel_encoder = rel_encoder
        self.tone_cols = tone_cols
        self.rel_cols = rel_cols
        self.predict_tones = predict_tones_fn
        self.predict_rel = predict_rel_fn

        for col in self.tone_cols + self.rel_cols:
            if col not in self.wishes_df.columns:
                self.wishes_df[col] = 0

        self.wishes_df['lemmatized_quote'] = self.wishes_df['quote'].apply(preprocess_input)
        self.wishes_df['tfidf'] = list(self.vectorizer.transform(self.wishes_df['lemmatized_quote']).toarray())

    def recommend(self, user_text, tone_weight=0.3, tfidf_weight=0.7):
        rel = self.predict_rel(user_text)
        tones = self.predict_tones(user_text)

        filtered = self.wishes_df[self.wishes_df[rel] == 1].copy()
        if filtered.empty:
            return "No wishes found for that relationship."

        user_vec = self.vectorizer.transform([preprocess_input(user_text)]).toarray()[0]
        best_score = float('inf')
        best_wish = "Hope you have an amazing birthday!"

        for _, row in filtered.iterrows():
            tone_match = sum([row[t] for t in tones if t in self.tone_cols])
            tone_score = 1 - (tone_match / len(tones)) if tones else 1
            tfidf_sim = cosine_similarity([user_vec], [row['tfidf']])[0][0]
            tfidf_dist = 1 - tfidf_sim
            score = (tone_weight * tone_score) + (tfidf_weight * tfidf_dist)
            if score < best_score:
                best_score = score
                best_wish = row['quote']

        return best_wish
