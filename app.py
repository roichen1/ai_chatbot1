from flask import Flask, request, jsonify, render_template, session
import os
import requests

app = Flask(__name__, template_folder="templates")
app.secret_key = "supersecretkey"

# Load Hugging Face API Key from environment variables
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Hugging Face Inference API endpoint for text generation
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Function to query the Hugging Face Inference API
def query_huggingface(prompt):
    try:
        print(f"📨 Sending request to Hugging Face API with prompt: {prompt}")
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            bot_reply = data[0]["generated_text"]
        else:
            bot_reply = "Sorry, I couldn't generate a response."
        print(f"📩 Hugging Face API response: {bot_reply}")
        return bot_reply
    except Exception as e:
        print(f"❌ Error in query_huggingface(): {e}")
        return "Error processing request"

@app.route('/')
def home():
    session.clear()
    return render_template('index.html')

@app.route('/start_chat', methods=['POST'])
def start_chat():
    user_topic = request.json.get("topic")
    if not user_topic:
        return jsonify({"error": "Please enter a topic to discuss"}), 400
    
    session['chat_history'] = []
    prompt = f"You are a helpful assistant that supports the user in a conversation about {user_topic}. Try to ask insightful questions and help the user gain new insights."
    session['chat_history'].append({"role": "system", "content": prompt})
    
    return jsonify({"message": "Chat started! What would you like to know about the topic?"})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Please enter a message"}), 400

    print(f"📨 Received user input: {user_input}")
    session['chat_history'].append({"role": "user", "content": user_input})
    context = " ".join([m["content"] for m in session['chat_history']])
    
    bot_reply = query_huggingface(context)  # Use API-based response generation
    
    session['chat_history'].append({"role": "assistant", "content": bot_reply})
    
    return jsonify({"message": bot_reply})

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_type = request.json.get("feedback")
    if feedback_type not in ["like", "dislike"]:
        return jsonify({"error": "Invalid feedback"}), 400
    return jsonify({"message": "Thank you for your feedback!"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if no PORT is set
    app.run(host="0.0.0.0", port=port, debug=True)
