from flask import Flask, request, jsonify
from flask_cors import CORS
from Augmentation import BirthdayWishAugmentor
import Nlp

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS from any origin (for dev)

# Initialize your augmentor with NLP models and resources
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

# API Endpoint to generate a birthday wish
@app.route('/generate-wish', methods=['POST'])
def generate_wish():
    data = request.get_json()

    input_text = data.get('text')
    if not input_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        wish = augmentor.recommend(input_text)
        return jsonify({'wish': wish})
    except Exception as e:
        print(f"[ERROR] Failed to generate wish: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
