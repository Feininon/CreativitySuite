import base64
import os
import uuid
import re
import json
import requests
from flask import Flask, render_template, request, jsonify

# NEW: Import torch and the diffusers pipeline
import torch
from diffusers import AutoPipelineForText2Image

# --- Configuration ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"

app = Flask(__name__)

# --- NEW: Load Stable Diffusion Model on Startup ---
# We load the model once when the app starts to avoid reloading it on every request.
# This will be slow the first time, but fast for subsequent generations.
print("Loading Stable Diffusion model... This may take a while and require a lot of RAM/VRAM.")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", # A fast model good for quick results
    torch_dtype=torch.float16, 
    variant="fp16"
)
# Send the model to the GPU if available (recommended)
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
print("✅ Stable Diffusion model loaded.")


# --- NEW: Image Generation function using diffusers ---
def generate_image_with_diffusers(prompt, panel_filename):
    """
    Generates an image using the loaded diffusers pipeline and saves it.
    """
    # A good starting prompt structure for comics
    full_prompt = f"comic book panel, {prompt}, vibrant color, digital art, clean lines"
    
    try:
        # Generate the image. `num_inference_steps` is low for a turbo model.
        image = pipe(prompt=full_prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        
        # Define the path to save the image
        output_path = os.path.join('static', 'generated_images', panel_filename)
        
        # Save the image
        image.save(output_path)
        
        # Return the web-accessible path for the <img> src attribute
        return f"/{output_path}"

    except Exception as e:
        print(f"Error generating image with diffusers: {e}")
        return None # Indicate failure


# --- Ollama functions (no changes needed here) ---
def generate_art_from_ollama(prompt, art_type):
    # This function for SVG/ASCII remains unchanged
    # ... (code is the same as before)
    if art_type == 'svg':
        system_prompt = "You are an expert SVG artist. Your sole purpose is to generate valid, single-file SVG code based on a user's description. Do not include any explanation, preamble, or markdown code fences like ```svg. Only output the raw <svg>...</svg> code. The SVG should be self-contained and not link to external files. Make it visually appealing."
        full_prompt = f"Create an SVG for: {prompt}"
    elif art_type == 'ascii':
        system_prompt = "You are a master of ASCII art. Your only job is to convert the user's text description into detailed, complex, and beautiful ASCII art. Use a wide variety of characters to create shading and texture. Do not provide any commentary, explanation, or markdown fences like ```. Just output the raw ASCII art."
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
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return {"error": "Could not connect to the Ollama API. Make sure Ollama is running."}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": "An unexpected error occurred."}


# --- Page Routes (no changes needed here) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/svg')
def svg_page():
    return render_template('svg_generator.html')

@app.route('/ascii')
def ascii_page():
    return render_template('ascii_generator.html')

@app.route('/comic')
def comic_page():
    return render_template('comic_generator.html')

# --- API Endpoints ---
@app.route('/generate', methods=['POST'])
def generate():
    # This endpoint for SVG/ASCII remains unchanged
    data = request.get_json()
    prompt = data.get('prompt')
    art_type = data.get('type')
    if not prompt or not art_type:
        return jsonify({"error": "Prompt and type are required."}), 400
    result = generate_art_from_ollama(prompt, art_type)
    return jsonify(result)


# --- MODIFIED: Comic Generation Pipeline ---
@app.route('/generate-comic', methods=['POST'])
def generate_comic_pipeline():
    # Step 1: Generate script (this part is the same)
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt: return jsonify({"error": "A prompt is required."}), 400
    print("Step 1: Generating comic script...")
    script_system_prompt = "..." # Same long prompt as before
    try:
        # ... (Same code to call Ollama and get the JSON script) ...
        script_response = requests.post(OLLAMA_ENDPOINT, json={ "model": OLLAMA_MODEL, "prompt": f"Create a comic script for this idea: {prompt}", "system": script_system_prompt, "format": "json", "stream": False})
        script_response.raise_for_status()
        response_text = script_response.json().get("response", "{}")
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match: return jsonify({"error": "LLM did not return valid JSON for the script."}), 500
        comic_data = json.loads(json_match.group(0))
    except Exception as e:
        return jsonify({"error": f"Failed to generate comic script: {e}"}), 500

    print("Step 2: Generating images for each panel with diffusers...")
    # Step 2: Generate Images (this part is modified)
    try:
        os.makedirs(os.path.join('static', 'generated_images'), exist_ok=True)
        for page in comic_data.get("pages", []):
            for panel in page.get("panels", []):
                description = panel.get("description", "A blank panel.")
                page_num, panel_num = page.get('page'), panel.get('panel')
                print(f"  - Generating image for Page {page_num}, Panel {panel_num}...")
                filename = f"comic_{uuid.uuid4().hex[:8]}_p{page_num}_n{panel_num}.png"
                
                # *** THIS IS THE KEY CHANGE ***
                # Call our new internal function instead of an external API
                image_path = generate_image_with_diffusers(description, filename)
                
                panel["image_url"] = image_path
    except Exception as e:
        print(f"Error generating panel art: {e}")
        return jsonify({"error": f"Failed during panel art generation: {e}"}), 500

    print("Step 3: Comic generation complete.")
    return jsonify(comic_data)

if __name__ == '__main__':
    app.run(debug=True)