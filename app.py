from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Бағаны болжау моделі
X_reg = [[60], [70], [80], [90], [100]]
y_reg = [3, 3, 4, 4, 5]

reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

# Үлгерімді анықтау моделі
X_cls = [[60], [75], [85], [95]]
y_cls = ["Қанағаттанарлық", "Жақсы", "Жақсы", "Үздік"]

cls_model = DecisionTreeClassifier()
cls_model.fit(X_cls, y_cls)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():

    message = request.form["message"].lower()

    if "математика" in message:
        answer = "Математика – сандар мен есептеулер туралы ғылым."

    elif "информатика" in message:
        answer = "Информатика – ақпаратты өңдеу және бағдарламалау ғылымы."

    elif "пифагор" in message:
        answer = "Пифагор теоремасы: a² + b² = c²"

    else:
        answer = "Кешіріңіз, бұл сұраққа жауап табылмады."

    return render_template(
        "index.html",
        user_message=message,
        answer=answer
    )

@app.route("/predict", methods=["POST"])
def predict():

    attendance = int(request.form["attendance"])

    grade = reg_model.predict([[attendance]])

    return render_template(
        "index.html",
        grade=round(grade[0], 1)
    )

@app.route("/classify", methods=["POST"])
def classify():

    score = int(request.form["score"])

    result = cls_model.predict([[score]])

    return render_template(
        "index.html",
        level=result[0]
    )

if __name__ == "__main__":
    app.run(debug=True)