from flask import Flask, request, jsonify, render_template, session
import openai  # ניתן להחליף ב- API חינמי אחר אם אין לך גישה חינמית

app = Flask(__name__)
app.secret_key = "supersecretkey"  # מפתח לשמירת הסשן של המשתמשים

# קביעת מפתח ה-API שלך
openai.api_key = "your_openai_api_key"  # יש להחליף במפתח שלך

@app.route('/')
def home():
    session.clear()  # מאפס את השיחה עם כניסה חדשה
    return render_template('index.html')

@app.route('/start_chat', methods=['POST'])
def start_chat():
    user_topic = request.json.get("topic")
    if not user_topic:
        return jsonify({"error": "אנא הזן נושא לשיחה"}), 400
    
    session['chat_history'] = []  # יצירת היסטוריית שיחה חדשה
    
    prompt = f"""
    אתה עוזר חכם שתומך במשתמש בשיחה על {user_topic}.
    נסה לשאול שאלות מעמיקות ולעזור למשתמש לקבל תובנות חדשות. ענה בצורה קצרה וברורה.
    """
    
    session['chat_history'].append({"role": "system", "content": prompt})
    return jsonify({"message": "שיחה התחילה! מה תרצה לדעת על הנושא?"})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "אנא הכנס הודעה"}), 400
    
    session['chat_history'].append({"role": "user", "content": user_input})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # ניתן להחליף במודל חינמי אחר
        messages=session['chat_history']
    )
    bot_reply = response["choices"][0]["message"]["content"]
    
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
