import pandas as pd
import numpy as np
import yaml
import random
from sklearn.metrics.pairwise import cosine_similarity


class BirthdayWishAugmentor:
    def __init__(self, wishes_path, yaml_path, vectorizer, model_tone, model_rel, rel_encoder,
                 tone_cols, rel_cols, predict_tones_fn, predict_rel_fn):
        # Load wishes dataset
        self.wishes_df = pd.read_csv(wishes_path).fillna("")
        self.yaml_path = yaml_path
        self.vectorizer = vectorizer
        self.model_tone = model_tone
        self.model_rel = model_rel
        self.rel_encoder = rel_encoder
        self.tone_cols = tone_cols
        self.rel_cols = rel_cols
        self.predict_tones = predict_tones_fn
        self.predict_rel = predict_rel_fn

        # Build synonym map
        self.synonym_map = self._load_and_build_synonyms()

        # Ensure tone and rel columns exist
        for col in self.tone_cols:
            if col not in self.wishes_df.columns:
                self.wishes_df[col] = 0
        for col in self.rel_cols:
            if col not in self.wishes_df.columns:
                self.wishes_df[col] = 0

        # Add TF-IDF vectors
        self.wishes_df['tfidf'] = list(self.vectorizer.transform(self.wishes_df['quote']).toarray())

    def _load_and_build_synonyms(self):
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        birthday_synonyms = data.get("Birthday_Synonyms", [])
        relationship_synonyms = data.get("Relationship_Synonyms", [])
        combined = birthday_synonyms + relationship_synonyms

        mapping = {}
        for group in combined:
            if isinstance(group, list):
                group = [w for w in group if isinstance(w, str)]
                for word in group:
                    mapping[word.lower()] = [w for w in group if w.lower() != word.lower()]
        return mapping

    def _augment_text_with_synonyms(self, text, min_replacements=1):
        tokens = text.split()
        indexed_tokens = []
        for i, token in enumerate(tokens):
            clean = token.strip(",.!?;:\"'()[]{}").lower()
            if clean in self.synonym_map:
                indexed_tokens.append((i, clean))

        if not indexed_tokens:
            return text

        num_to_replace = random.randint(min_replacements, len(indexed_tokens))
        indices_to_replace = random.sample(indexed_tokens, k=num_to_replace)

        for i, word in indices_to_replace:
            replacement = random.choice(self.synonym_map[word])
            if tokens[i][0].isupper():
                replacement = replacement.capitalize()
            punct = ''.join([c for c in tokens[i] if not c.isalnum()])
            tokens[i] = replacement + punct

        return " ".join(tokens)

    def recommend(self, user_text, tone_weight=0.7, tfidf_weight=0.3):
        predicted_relationship = self.predict_rel(user_text)
        filtered_df = self.wishes_df[self.wishes_df[predicted_relationship] == 1]

        if filtered_df.empty:
            return "No wishes found for the predicted relationship."

        detected_tones = self.predict_tones(user_text)
        print(f"🎭 Predicted Tone(s): {', '.join(detected_tones)}")

        best_score = float('inf')
        best_wish = "Wishing you a fantastic birthday!"

        tfidf_vec = self.vectorizer.transform([user_text]).toarray()[0]

        for _, row in filtered_df.iterrows():
            wish_tone = row[self.tone_cols].values.astype(int)
            dist_tone = sum([1 for tone in detected_tones if tone not in wish_tone])
            sim = cosine_similarity([tfidf_vec], [row['tfidf']])[0][0]
            tfidf_dist = 1 - sim

            score = (tone_weight * dist_tone) + (tfidf_weight * tfidf_dist)

            if score < best_score:
                best_score = score
                best_wish = row['quote']

        print(f"\n🎁 Wish Found: {best_wish}")
        best_wish_aug = self._augment_text_with_synonyms(best_wish)

        # White-box info
        print(f"\n🔎 User Request: {user_text}")
        print(f"💬 Predicted Relationship: {predicted_relationship}")
        print(f"🎭 Predicted Tone(s): {', '.join(detected_tones)}")
        print(f"\n🎁 Final Wish: {best_wish_aug}")

        return best_wish_aug
