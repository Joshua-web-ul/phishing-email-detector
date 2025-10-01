from flask import Flask, request, render_template
from src.models.logistic_regression import LogisticRegressionModel
from src.utils.explain_why import explain_why

app = Flask(__name__)

logistic_model = LogisticRegressionModel()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email_text = request.form['email_text']
        prediction = logistic_model.predict([email_text])[0]
        result = 'Phishing' if prediction == 1 else 'Legitimate'
        return render_template('result.html', email_text=email_text, result=result)
    return render_template('index.html')

@app.route('/explain', methods=['POST'])
def explain():
    email_text = request.form['email_text']
    explanation = explain_why(email_text)
    return render_template('explain.html', email_text=email_text, explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True)
