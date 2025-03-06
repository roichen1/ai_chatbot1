import os
import requests
import logging
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),  # Log to a file
            logging.StreamHandler()  # Also log to console
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Initialize Flask app
app = Flask(__name__)

# Hugging Face API configuration
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def check_api_key():
    """Verify the API key is available"""
    if not HUGGINGFACE_API_KEY:
        logger.error("HUGGINGFACE_API_KEY is NOT set or is empty!")
        return False
    else:
        logger.info(f"HUGGINGFACE_API_KEY loaded. Length: {len(HUGGINGFACE_API_KEY)} chars")
        return True

def query_llama(prompt):
    """
    Sends a request to Hugging Face's API to generate a response.
    
    Args:
        prompt (str): The user's input prompt
        
    Returns:
        str: The generated response or error message
    """
    # Log the input prompt (without revealing sensitive information)
    logger.info(f"Received prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Received prompt: {prompt}")
    
    if not check_api_key():
        return "API key is missing. Please set the HUGGINGFACE_API_KEY environment variable."
    
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    data = {"inputs": prompt}

    try:
        # Log the API request
        logger.info(f"Sending request to Hugging Face API: {HUGGINGFACE_API_URL}")
        
        response = requests.post(
            HUGGINGFACE_API_URL, 
            headers=headers, 
            json=data,
            timeout=30  # Add timeout to prevent hanging requests
        )
        
        # Log the API response status
        logger.info(f"Received response from Hugging Face API. Status code: {response.status_code}")

        if response.status_code == 200:
            try:
                response_data = response.json()
                generated_text = response_data[0]["generated_text"]
                # Log a preview of the generated response
                logger.info(f"Successfully generated response: {generated_text[:100]}...")  # Log first 100 chars
                return generated_text
            except (IndexError, KeyError) as e:
                error_msg = f"Unexpected response format: {str(e)}"
                logger.error(error_msg)
                return f"Error processing response: {error_msg}"
        else:
            # Log error response
            error_msg = f"Error: {response.status_code}, {response.text[:200]}"
            logger.error(error_msg)
            return f"API Error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        error_msg = "Request to Hugging Face API timed out"
        logger.error(error_msg)
        return error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Connection error when contacting Hugging Face API"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        # Log any exceptions
        error_msg = f"Exception in Hugging Face API call: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"An unexpected error occurred: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def home():
    """
    Serves the main chat interface and handles chat messages.
    """
    if request.method == "POST":
        logger.info("Received POST request at root endpoint")

        user_input = None

        # Check if the request is JSON
        if request.is_json:
            data = request.get_json(silent=True)
            if data and "message" in data:
                user_input = data.get("message", "")
        
        # Check if the request is Form Data
        else:
            user_input = request.form.get("user_input", "")
            logger.info("Form data detected")

        # If no valid input was received
        if not user_input:
            logger.warning("No valid input received")
            return jsonify({"error": "No message received"}), 400

        try:
            # Query Llama model and return response
            response = query_llama(user_input)
            return jsonify({"response": response})
        
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500

    logger.info("Rendering home page")
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Dedicated API endpoint for chat functionality
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400
    
    user_input = data["message"]
    response = query_llama(user_input)
    
    return jsonify({"response": response})

def main():
    """Main entry point for the application"""
    logger.info("Starting Flask application")
    
    # Get port from environment variable with a default value
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Will run on host 0.0.0.0, port {port}")
    
    # Only enable debug mode in development
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    # Start the application
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

if __name__ == "__main__":
    main()
