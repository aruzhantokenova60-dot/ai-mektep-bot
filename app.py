import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# --- 2-МОДУЛЬ: Деректер ғылымы/ML үшін математика және статистика ---
# Scikit-learn модельдері
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# --- 6-МОДУЛЬ: Ансамбльдер және күшейту (XGBoost интеграциясы) ---
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

app = Flask(__name__)

# =====================================================================
# ДЕРЕКТЕР ҚОРЫ МЕН МОДЕЛДЕРДІ ДАЙЫНДАУ (3, 4, 5, 6-МОДУЛЬДЕР)
# =====================================================================

# 2-МОДУЛЬ және 3-МОДУЛЬ: Статистикалық деректер жиынтығын дайындау (Pandas арқылы)
# Белгілер (Features): [Сабаққа қатысу %, Ағымдағы балл (100-ден)]
X_train_raw = [[60, 50], [70, 65], [80, 75], [90, 85], [100, 95], [50, 45], [85, 60]]
y_reg_train = [3.0, 3.5, 4.0, 4.5, 5.0, 2.5, 3.8] # 5 баллдық жүйемен болжамды баға (Регрессия)
y_cls_train = [1, 0, 0, 0, 0, 1, 0] # 1: Тәуекел тобы (үлгермейді), 0: Тұрақты (Жіктеу)

# 5-МОДУЛЬ: Машиналық оқытудағы регрессиялық модельдер (Сызықтық регрессия)
reg_model = LinearRegression()
reg_model.fit(X_train_raw, y_reg_train)

# 6-МОДУЛЬ: Машиналық оқытудағы жіктеу модельдері (Ансамбльдер және Күшейту)
# Біз бірнеше ансамбльдік модельдерді дайындап, салыстырамыз (3-модуль визуализациясы үшін)
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42), # Бэггинг
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),        # Күшейту
    "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42) # Градиентті күшейту
}

# Барлық модельдерді оқыту
for name, model in models.items():
    model.fit(X_train_raw, y_cls_train)

# Егер XGBoost орнатылған болса, оны да қосамыз
if XGB_AVAILABLE:
    xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
    xgb_model.fit(np.array(X_train_raw), np.array(y_cls_train))

# =====================================================================
# ВЕБ-МАРШРУТТАР (ROUTES) ЖӘНЕ API
# =====================================================================

@app.route("/")
def home():
    # 1, 9, 10-МОДУЛЬДЕР: Басты бет интерфейсін шақыру
    return render_template("index.html")


# 1, 8, 10-МОДУЛЬДЕР: NLP, Трансформаторлар және Этикалық Чат-бот
@app.route("/chat", methods=["POST"])
def chat():
    message = request.form.get("message", "").strip()
    message_lower = message.lower()
    
    # 10-МОДУЛЬ: Білім беруде ЖИ-ді этикалық пайдалану (Қауіпсіздік және плагиат сүзгісі)
    ethical_restrictions = ["көшіріп алу", "шпаргалка", "дайын эссе", "басқаның орнына", "алдау"]
    toxic_words = ["жаман", "ақымақ", "жек көрем"] # Этикаға жатпайтын лексика
    
    if any(word in message_lower for word in ethical_restrictions):
        answer = ("⚠️ **Академиялық адалдық ережесі:** Мен сізге дайын жауапты көшіріп алуға "
                  "немесе академиялық қулықтар жасауға көмектесе алмаймын. Бірақ тақырыптың "
                  "мағынасын түсіндіріп беруге дайынмын!")
        return render_template("index.html", user_message=message, answer=answer)
        
    if any(word in message_lower for word in toxic_words):
        answer = "🛑 **Этикалық кодекс:** Өтінемін, чат-ботпен сөйлесу кезінде өзара сыйластық пен сыпайылық танытыңыз."
        return render_template("index.html", user_message=message, answer=answer)

    # 1-МОДУЛЬ (Үрдістер) және 8-МОДУЛЬ (NLP және Ережеге негізделген интеллектуалды ассистент)
    if "математика" in message_lower:
        answer = "🧮 **Математика:** Сандар, құрылымдар және кеңістік туралы ғылым. 2-модульде біз оның ML-ге арналған статистика бөлімін өттік."
    elif "информатика" in message_lower:
        answer = "💻 **Информатика:** Ақпаратты жинау, сақтау және бағдарламалық құралдармен (мысалы, осы Flask қолданбасымен) өңдеу ғылымы."
    elif "пифагор" in message_lower:
        answer = "📐 **Пифагор теоремасы:** Тік бұрышты үшбұрышта гипотенузаның квадраты катеттерінің квадраттарының қосындысына тең: a² + b² = c²."
    elif "ансамбль" in message_lower or "бустинг" in message_lower:
        answer = "🌲 **6-модуль анықтамасы:** Ансамбльдер — бұл күшті болжам жасау үшін бірнеше әлсіз модельдерді (мысалы, жүздеген шешім ағаштарын) біріктіру әдісі."
    elif "нейрондық желі" in message_lower or "компьютерлік көру" in message_lower:
        answer = "👁️ **7-модуль анықтамасы:** Терең оқыту (Deep Learning) және CNN адамның көру жүйесі сияқты суреттерді тануға және нысандарды жіктеуге мүмкіндік береді."
    else:
        # 8-МОДУЛЬ: Генеративті ЖИ стиліндегі балама жауап
        answer = f"🤖 Мен интеллектуалды мектеп көмекшісімін. Сіздің сұрағыңыз қабылданды: '{message}'. Оқу бағдарламасы бойынша көмектесуге дайынмын!"

    return render_template("index.html", user_message=message, answer=answer)


# 5-МОДУЛЬ: Машиналық оқытудағы регрессиялық модельдер (Бағаны және ҰБТ-ны болжау)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        attendance = int(request.form["attendance"])
        current_score = int(request.form.get("current_score", 70)) # Әдепкі бойынша 70 балл
        
        # Кіріс деректерін екі белгі бойынша жинау [Қатысу, Балл]
        features = [[attendance, current_score]]
        grade = reg_model.predict(features)
        
        # Бағаны 2 мен 5 аралығында шектеу (математикалық түзету)
        final_grade = max(2.0, min(5.0, round(grade[0], 1)))
        
        return render_template("index.html", grade=final_grade, attendance=attendance, current_score=current_score)
    except Exception as e:
        return render_template("index.html", error=f"Регрессиялық есептеу қатесі: {str(e)}")


# 6-МОДУЛЬ: Ансамбльдік модельдер арқылы оқушыны жіктеу (Classification)
@app.route("/classify", methods=["POST"])
def classify():
    try:
        score = int(request.form["score"])
        attendance = int(request.form.get("attendance_cls", 80)) # Әдепкі бойынша 80% қатысу
        model_type = request.form.get("model_type", "RandomForest") # Пайдаланушы таңдаған ансамбль
        
        features = [[attendance, score]]
        
        # Таңдалған ансамбльдік модель бойынша болжам жасау
        if model_type == "XGBoost" and XGB_AVAILABLE:
            prediction = xgb_model.predict(np.array(features))[0]
        elif model_type in models:
            prediction = models[model_type].predict(features)[0]
        else:
            prediction = models["RandomForest"].predict(features)[0] # Default
            
        # 1: Тәуекел тобы, 0: Тұрақты оқушы
        result_text = "🔴 Тәуекел тобы (Қосымша көмек қажет!)" if prediction == 1 else "🟢 Тұрақты (Үлгерімі жақсы)"
        
        return render_template("index.html", level=result_text, score=score, selected_model=model_type)
    except Exception as e:
        return render_template("index.html", error=f"Жіктеу модель қатесі: {str(e)}")


# 7-МОДУЛЬ: Компьютерлік көру және Нысандарды тану (Студент формасын/ID картасын симуляциялау)
@app.route("/vision_analyze", methods=["POST"])
def vision_analyze():
    # Мектеп қауіпсіздігі үшін CNN желілері мен OpenCV жұмысын имитациялайтын API
    # Нақты камера болмаса да, нейрондық желінің жұмыс нәтижесін көрсетеді
    recognition_results = {
        "status": "Сәтті аяқталды",
        "detected_object": "Мектеп формасы / Оқушы ID- картасы",
        "confidence": "98.4% (CNN дәлдігі)",
        "access": "Рұқсат берілді"
    }
    return render_template("index.html", vision=recognition_results)

if __name__ == "__main__":
    # 9-МОДУЛЬ: Тұрақты локальді серверде іске қосу
    app.run(debug=True)
