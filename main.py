# todo must be python 3.10 and not latest version
from Augmentation import BirthdayWishAugmentor
import Nlp

# Use the vectorizer, model_tone, model_rel, rel_encoder from Nlp.py
augmentor = BirthdayWishAugmentor(
    wishes_path='big_dataset.csv',
    yaml_path='Synonym_Dictionary (2).yaml',
    vectorizer=Nlp.vectorizer,
    model_tone=Nlp.model_tone,
    model_rel=Nlp.model_rel,
    rel_encoder=Nlp.rel_encoder,
    tone_cols=Nlp.tone_cols,
    rel_cols=Nlp.rel_cols,
    predict_tones_fn=Nlp.predict_tones,
    predict_rel_fn=Nlp.predict_rel
)

# Try it out
sample_inputs = [
    "Write a birthday wish for my wife that's happy and rhyming",
    "Something professional and short for my boss",
    "Say something to my son that's funny and inspirational"
]

for text in sample_inputs:
    print("\n" + "=" * 70)
    augmentor.recommend(text)
