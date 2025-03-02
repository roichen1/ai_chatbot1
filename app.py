from flask import Flask, request, jsonify, render_template, session
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# הגדרת ה-API KEY מהסביבה
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # שמור את הטוקן כמשתנה סביבה

# הגדרת המודל של LLaMA
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # ניתן להחליף למודל אחר
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_API_KEY)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", use_auth_token=HF_API_KEY
)

# פונקציה לשליחת טקסט למודל
def query_llama(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/')
def home():
    session.clear()
    return render_template('index.html')

@app.route('/start_chat', methods=['POST'])
def start_chat():
    user_topic = request.json.get("topic")
    if not user_topic:
        return jsonify({"error": "אנא הזן נושא לשיחה"}), 400
    
    session['chat_history'] = []
    prompt = f"אתה עוזר חכם שתומך במשתמש בשיחה על {user_topic}. נסה לשאול שאלות מעמיקות ולעזור למשתמש לקבל תובנות חדשות."
    session['chat_history'].append({"role": "system", "content": prompt})

    return jsonify({"message": "שיחה התחילה! מה תרצה לדעת על הנושא?"})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "אנא הכנס הודעה"}), 400

    session['chat_history'].append({"role": "user", "content": user_input})
    context = " ".join([m["content"] for m in session['chat_history']])

    bot_reply = query_llama(context)

    session['chat_history'].append({"role": "assistant", "content": bot_reply})

    return jsonify({"message": bot_reply})

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_type = request.json.get("feedback")
    if feedback_type not in ["like", "dislike"]:
        return jsonify({"error": "משוב לא חוקי"}), 400
    return jsonify({"message": "תודה על המשוב!"})

if __name__ == '__main__':
    app.run(debug=True)
