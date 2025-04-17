from flask import Flask, request, jsonify
from flask_cors import CORS
from Augmentation import BirthdayWishAugmentor
import Nlp

app = Flask(__name__)
CORS(app)

# Initialize the BirthdayWishAugmentor with the required resources
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
    # Parse the incoming JSON request
    data = request.get_json()
    print(f"Received request: {data}")  # Log the incoming request
    text = data.get('text', '')  # User input text
    blacklist_ids = data.get('blacklist_ids', [])  # IDs to exclude

    if not text:
        return jsonify({'error': 'No input provided'}), 400

    try:
        # Generate a wish while excluding blacklisted IDs
        result = augmentor.recommend(text, blacklist_ids=blacklist_ids)
        print(f"The new wish is: {result}")
        return jsonify(result)  # <-- This now returns both 'wish' and 'wish_id'
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': 'Internal error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
