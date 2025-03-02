from flask import Flask, request, jsonify, render_template, session
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load the Hugging Face API key from the environment variables
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Load the LLaMA model from Hugging Face
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_API_KEY)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", use_auth_token=HF_API_KEY
)

# Function to send queries to the LLaMA model
def query_llama(prompt):
    try:
        print(f"📨 Sending request to LLaMA model with prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(**inputs, max_length=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"📩 LLaMA model response: {response}")
        return response
    except Exception as e:
        print(f"❌ Error in query_llama(): {e}")
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
    
    bot_reply = query_llama(context)
    
    session['chat_history'].append({"role": "assistant", "content": bot_reply})
    
    return jsonify({"message": bot_reply})

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_type = request.json.get("feedback")
    if feedback_type not in ["like", "dislike"]:
        return jsonify({"error": "Invalid feedback"}), 400
    return jsonify({"message": "Thank you for your feedback!"})

if __name__ == '__main__':
    app.run(debug=True)
