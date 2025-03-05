import os
import requests
import logging
from flask import Flask, request, jsonify, render_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Log to a file
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Hugging Face API Endpoint
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Set this in Render Environment Variables

def query_llama(prompt):
    """
    Sends a request to Hugging Face's API to generate a response with detailed logging.
    """
    # Log the input prompt
    logger.info(f"Received prompt: {prompt}")
    
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    data = {"inputs": prompt}

    try:
        # Log the API request details
        logger.info(f"Sending request to Hugging Face API: {HUGGINGFACE_API_URL}")
        
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=data)

        # Log the API response status
        logger.info(f"Received response from Hugging Face API. Status code: {response.status_code}")

        if response.status_code == 200:
            generated_text = response.json()[0]["generated_text"]
            # Log the generated response
            logger.info(f"Successfully generated response: {generated_text[:100]}...")  # Log first 100 chars
            return generated_text
        else:
            # Log error response
            error_msg = f"Error: {response.status_code}, {response.text}"
            logger.error(error_msg)
            return error_msg
    except Exception as e:
        # Log any exceptions
        error_msg = f"Exception in Hugging Face API call: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg

@app.route("/", methods=["GET", "POST"])
def home():
    """
    Serves the main chat interface and handles chat messages directly.
    """
    if request.method == "POST":
        logger.info("Received POST request at root endpoint")

        user_input = None

        # Check if the request is JSON
        if request.content_type == "application/json":
            data = request.get_json(silent=True)
            if data and "message" in data:
                user_input = data["message"]
        
        # Check if the request is Form Data (multipart/form-data or x-www-form-urlencoded)
        elif request.content_type.startswith("multipart/form-data") or request.content_type == "application/x-www-form-urlencoded":
            logger.info("form data detected")
            user_input = request.form.get("user_input", "")
            logger.info(f"user_input: {user_input}")

        # If no valid input was received
        if not user_input:
            logger.info("no valid input recieved")
            return jsonify({"error": "No message received"}), 400

        try:
            # Query Llama model and return response
            response = query_llama(user_input)
            logger.info(f"Response from Llama: {response}")

            return jsonify({"response": response})
        
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500

    logger.info("Rendering home page")
    return render_template("index.html")

if __name__ == "__main__":
    # Log app startup
    logger.info("Starting Flask application")
    
    # Render requires 0.0.0.0 and a dynamic port
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Will run on host 0.0.0.0, port {port}")
    
    app.run(host="0.0.0.0", port=port, debug=True)
