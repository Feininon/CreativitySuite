
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import random # For random selection
import time   # For the delay

app = Flask(__name__)

# --- Load Models and Databases on Startup ---
print("Loading embedding model and art databases...")

try:
    retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load SVG database
    svg_embeddings = np.load('svg_embeddings.npy')
    with open('svg_library.json', 'r') as f:
        svg_library = json.load(f)
    print("✅ SVG database loaded.")

    # Load ASCII database
    ascii_embeddings = np.load('ascii_embeddings.npy')
    with open('ascii_library.json', 'r') as f:
        ascii_library = json.load(f)
    print("✅ ASCII database loaded.")

except FileNotFoundError:
    print("\n❌ ERROR: Database files not found.")
    print("Please run `python build_databases.py` first to create the necessary files.")
    exit()

# --- Page Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/svg-retriever')
def svg_retriever_page():
    return render_template('retriever_svg.html')

@app.route('/ascii-retriever')
def ascii_retriever_page():
    return render_template('retriever_ascii.html')

# --- API Endpoint (MODIFIED LOGIC) ---
@app.route('/retrieve-art', methods=['POST'])
def retrieve_art():
    data = request.get_json()
    prompt = data.get('prompt')
    art_type = data.get('type')

    if not prompt or not art_type:
        return jsonify({"error": "Prompt and art type are required."}), 400

    if art_type == 'svg':
        library = svg_library
        embeddings = svg_embeddings
    elif art_type == 'ascii':
        library = ascii_library
        embeddings = ascii_embeddings
    else:
        return jsonify({"error": "Invalid art type."}), 400

    # 1. Generate an embedding for the user's prompt
    prompt_embedding = retrieval_model.encode([prompt])

    # 2. Calculate similarities
    similarities = 1 - cdist(prompt_embedding, embeddings, 'cosine')[0]
    
    # 3. Get the indices of the top 5 most similar items
    # argsort sorts from smallest to largest, so we take the last 5
    top_5_indices = np.argsort(similarities)[-5:]

    # 4. Randomly select one index from the top 5
    chosen_index = random.choice(top_5_indices)
    
    # 5. Retrieve the chosen art
    retrieved_art = library[chosen_index]

    # 6. Add a random delay between 10 and 20 seconds
    delay_duration = random.uniform(10, 20)
    print(f"Waiting for {delay_duration:.2f} seconds before responding...")
    time.sleep(delay_duration)

    return jsonify(retrieved_art)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

