from flask import Flask, request, jsonify
from flask_cors import CORS
from Augmentation import BirthdayWishAugmentor
import Nlp

app = Flask(__name__)
CORS(app)

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


@app.route('/generate-wish', methods=['POST'])
def generate_wish():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No input provided'}), 400
    try:
        result = augmentor.recommend(text)
        return jsonify({'wish': result})
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': 'Internal error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
