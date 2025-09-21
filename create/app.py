import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from diffusers import StableDiffusionPipeline
from scipy.spatial.distance import cdist
import time   
import re
import requests
import traceback
import uuid
import torch

# --- Configuration ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
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
# --- Lazy loading image model ---
image_pipeline = None

def get_image_pipeline():
    global image_pipeline
    if image_pipeline is None:
        print("🚀 Loading Waifu Diffusion (tiny comic/anime style) model...")
        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                "hakurei/waifu-diffusion",
                torch_dtype=torch.float32  # CPU safe, change to float16 if using GPU
            )
            # Uncomment below to use GPU
            #if torch.cuda.is_available():
             #    pipeline = pipeline.to("cuda")
            image_pipeline = pipeline
            print("✅ Model ready.")
        except Exception:
            print("❌ Failed to load Waifu Diffusion model:")
            traceback.print_exc()
            return None
    return image_pipeline

# --- Image generation ---
def generate_image_with_diffusers(prompt, panel_filename):
    pipe = get_image_pipeline()
    if pipe is None:
        print("⚠️ Cannot generate image: pipeline not available")
        return None

    full_prompt = f"comic book panel, {prompt}, cinematic, vibrant colors, clean lines, dynamic composition"
    try:
        image = pipe(prompt=full_prompt, num_inference_steps=25, guidance_scale=7.5).images[0]

        output_dir = os.path.join("static", "generated_images")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, panel_filename)

        image.save(output_path)
        return f"/{output_path}"

    except Exception:
        print("❌ Error generating image:")
        traceback.print_exc()
        return None

# --- Ollama art ---
def generate_art_from_ollama(prompt, art_type):
    if art_type == 'svg':
        system_prompt = "You are an expert SVG artist..."
        full_prompt = f"Create an SVG for: {prompt}"
    elif art_type == 'ascii':
        system_prompt = "You are a master of ASCII art..."
        full_prompt = f"Create complex ASCII art for: {prompt}"
    else:
        return {"error": "Invalid art type specified."}
    try:
        response = requests.post(
            OLLAMA_ENDPOINT,
            json={"model": OLLAMA_MODEL, "prompt": full_prompt, "system": system_prompt, "stream": False}
        )
        response.raise_for_status()
        generated_text = response.json().get("response", "").strip()
        return {"content": generated_text}
    except Exception:
        print("❌ Ollama generation error:")
        traceback.print_exc()
        return {"error": "Unexpected error during Ollama call"}

# --- Parse script into scenes ---
def parse_script_into_scenes(script_text):
    try:
        scenes = []
        current_scene = None

        for raw_line in script_text.splitlines():
            line = raw_line.strip()

            # Scene detection
            if re.match(r"(\*+)?\s*Scene\s+\d+[:\-]?.*", line, re.I):
                if current_scene:
                    scenes.append(current_scene)
                scene_name = re.sub(r"[\*\[\]]", "", line).strip()
                current_scene = {"scene": scene_name, "panels": []}

            # Panel detection
            elif re.search(r"Panel\s+\d+", line, re.I) and current_scene:
                clean_line = re.sub(r"[\*\[\]:]", "", line).strip()
                panel_match = re.match(r"Panel\s+(\d+)\s*(.*)", clean_line, re.I)
                if panel_match:
                    panel_num = panel_match.group(1)
                    panel_desc = panel_match.group(2).strip()
                    current_scene["panels"].append({
                        "panel": f"Panel {panel_num}",
                        "description": panel_desc
                    })

        if current_scene:
            scenes.append(current_scene)
        return scenes

    except Exception:
        print("❌ Error parsing script:")
        traceback.print_exc()
        return []
import random
# --- Build panel prompt for image model ---
def build_panel_prompt(description):
    return f"{description}, comic book style, dramatic lighting, vibrant colors, cinematic angle, clean digital ink lines, dynamic composition"

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

@app.route('/generate-comic', methods=['POST'])
def generate_comic_pipeline():
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({"error": "A prompt is required"}), 400

        print("Step 1: Generating comic script...")
        script_system_prompt = """You are a master comic book writer.
Write the story in Scenes and Panels format:
Scene 1:
Panel 1: ...
Panel 2: ...
Scene 2:
Panel 1: ...
"""

        script_response = requests.post(OLLAMA_ENDPOINT, json={
            "model": OLLAMA_MODEL,
            "prompt": f"Create a comic script for this idea: {prompt}",
            "system": script_system_prompt,
            "stream": False
        })
        script_response.raise_for_status()
        script_text = script_response.json().get("response", "").strip()
        print("Script preview:\n", script_text[:300], "...")

        print("Step 2: Parsing script into scenes...")
        scenes = parse_script_into_scenes(script_text)
        print(f"Parsed {len(scenes)} scenes.")

        print("Step 3: Generating images for each panel...")
        for scene in scenes:
            for idx, panel in enumerate(scene["panels"], start=1):
                print(f"  - Generating image for {scene['scene']} {panel['panel']} ...")
                filename = f"comic_{uuid.uuid4().hex[:8]}_{scene['scene'].replace(' ','_')}_p{idx}.png"
                rich_prompt = build_panel_prompt(panel["description"])
                print(f"    Prompt: {rich_prompt[:100]}...")  # debug first 100 chars
                try:
                    image_path = generate_image_with_diffusers(rich_prompt, filename)
                    panel["image_url"] = image_path
                except Exception:
                    print(f"❌ Error generating image for {scene['scene']} {panel['panel']}:")
                    traceback.print_exc()
                    panel["image_url"] = None

        print("✅ Comic generation complete.")
        return jsonify({"scenes": scenes})

    except Exception:
        print("❌ Error in /generate-comic pipeline:")
        traceback.print_exc()
        return jsonify({"error": "Unexpected error in /generate-comic"}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

