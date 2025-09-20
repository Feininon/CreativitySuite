import json
import requests
from flask import Flask, render_template, request, jsonify
import re

# --- Configuration ---
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:27b"

app = Flask(__name__)

# --- This function remains exactly the same ---
def generate_art_from_ollama(prompt, art_type):
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

# --- Page Routes ---

@app.route('/')
def index():
    """Renders the main menu page."""
    return render_template('index.html')

# NEW: Route for the SVG Generator page
@app.route('/svg')
def svg_page():
    """Renders the SVG generator page."""
    return render_template('svg_generator.html')

# NEW: Route for the ASCII Art Generator page
@app.route('/ascii')
def ascii_page():
    """Renders the ASCII Art generator page."""
    return render_template('ascii_generator.html')
@app.route('/comic')
def comic_page():
    """Renders the Comic Book generator page."""
    return render_template('comic_generator.html')


# --- New API Endpoint for the Comic Generation Pipeline ---

@app.route('/generate-comic', methods=['POST'])
def generate_comic_pipeline():
    """
    Handles the multi-step process of creating a comic.
    Step 1: Generate a JSON script for the comic.
    Step 2: For each panel in the script, generate SVG art.
    Step 3: Return the complete data structure.
    """
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "A prompt is required."}), 400

    print("Step 1: Generating comic script...")
    # === STEP 1: GENERATE THE COMIC SCRIPT ===
    script_system_prompt = """
    You are a comic book writer. Your task is to create a short comic book script based on a user's prompt.
    The output must be a valid JSON object. Do not include any text or markdown formatting before or after the JSON.
    The JSON structure should be:
    {
      "title": "Comic Title",
      "pages": [
        {
          "page": 1,
          "panels": [
            {
              "panel": 1,
              "description": "A detailed visual description of the scene for an artist to draw.",
              "caption": "A short narrative caption, or an empty string."
            },
            {
              "panel": 2,
              "description": "Visual description for the second panel.",
              "dialogue": "Character Name: 'The dialogue text.'"
            }
          ]
        }
      ]
    }
    Create a 3-page comic with 2-3 panels per page. Be creative and descriptive.
    """
    
    # First, we need to get the script from the LLM
    try:
        script_response = requests.post(
            OLLAMA_ENDPOINT,
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"Create a comic script for this idea: {prompt}",
                "system": script_system_prompt,
                "format": "json", # Ask Ollama to ensure the output is JSON
                "stream": False
            }
        )
        script_response.raise_for_status()
        
        # Ollama's JSON mode returns a string, so we need to parse it
        # Sometimes the model might still add extra text, so we clean it
        response_text = script_response.json().get("response", "{}")
        
        # A regex to find the JSON blob, just in case
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            return jsonify({"error": "LLM did not return valid JSON for the script."}), 500
        
        comic_data = json.loads(json_match.group(0))

    except Exception as e:
        print(f"Error generating comic script: {e}")
        return jsonify({"error": f"Failed to generate comic script: {e}"}), 500

    print("Step 2: Generating SVG for each panel...")
    # === STEP 2: GENERATE SVG FOR EACH PANEL ===
    try:
        for page in comic_data.get("pages", []):
            for panel in page.get("panels", []):
                description = panel.get("description", "A blank panel.")
                print(f"  - Generating art for Page {page.get('page')}, Panel {panel.get('panel')}...")
                
                # We reuse our existing SVG generation function!
                art_result = generate_art_from_ollama(description, 'svg')
                if "error" in art_result:
                    # If one panel fails, we can insert an error message or a blank
                    panel["svg"] = f'<svg width="100" height="100"><text x="10" y="50">Error generating image.</text></svg>'
                else:
                    panel["svg"] = art_result.get("content")
    
    except Exception as e:
        print(f"Error generating panel art: {e}")
        return jsonify({"error": f"Failed during panel art generation: {e}"}), 500

    print("Step 3: Comic generation complete.")
    # === STEP 3: RETURN THE COMPLETE COMIC DATA ===
    return jsonify(comic_data)


# --- API Endpoint (no changes here) ---

@app.route('/generate', methods=['POST'])
def generate():
    """The API endpoint to generate art."""
    data = request.get_json()
    prompt = data.get('prompt')
    art_type = data.get('type')
    if not prompt or not art_type:
        return jsonify({"error": "Prompt and type are required."}), 400
    result = generate_art_from_ollama(prompt, art_type)
    return jsonify(result)

if __name__ == '__main__':
    print("Starting the Creativity Suite backend...")
    print(f"Make sure Ollama is running and the model '{OLLAMA_MODEL}' is available.")
    app.run(debug=True)