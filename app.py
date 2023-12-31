from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import time

app = Flask(__name__)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define your API key
API_KEY = "abcd1234"

@app.route('/api/sentence/similarity', methods=['POST'])
def calculate_similarity():
    try:
        # Get input data from the POST request
        data = request.get_json()

        # Check if API key is provided and correct
        api_key = request.headers.get('Authorization')
        if api_key != API_KEY:
            return jsonify({'error': 'Incorrect API key'}), 401

        # Continue processing if API key is correct

        source_sentence = data['source_sentence']
        sentences = data.get('sentences', [])
        limit = data.get('limit', None)

        # Record start time
        start_time = time.time()

        # Encode source sentence
        emb_source = model.encode(source_sentence)

        # Limit the number of sentences processed
        sentences = sentences[:limit] if limit is not None else sentences

        # Encode input sentences and calculate Cosine Similarity
        results = []
        for sentence in sentences:
            emb = model.encode(sentence)
            cos_sim = util.cos_sim(emb_source, emb)
            results.append({
                'sentence': sentence,
                'similarity': cos_sim.item()
            })

        # Record end time
        end_time = time.time()

        # Calculate the time taken in milliseconds
        execution_time = (end_time - start_time) * 1000

        # Return the results with execution time as JSON
        result = {
            'source_sentence': source_sentence,
            'results': results,
            'execution_time': execution_time
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=1243)
