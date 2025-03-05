import os
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Hugging Face API Endpoint
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Set this in Render Environment Variables

def query_llama(prompt):
    """
    Sends a request to Hugging Face's API to generate a response.
    """
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    data = {"inputs": prompt}

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        return jsonify({"message": "POST request received at /"}), 200
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handles user chat input, queries the Llama API, and returns the response.
    """
    user_input = request.json.get("message", "")

    if not user_input:
        return jsonify({"error": "Empty input"}), 400

    response = query_llama(user_input)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    # Render requires 0.0.0.0 and a dynamic port
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
