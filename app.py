import os
from flask import Flask, request, render_template
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
import pandas as pd
emotions = [
    {
      "nombre": "admiración",
      "50": "Influencia",
      "30": "Percepción y Comprensión Emocional",
      "20": "Desarrollar y Estimular a los demás"
    },
    {
      "nombre": "alegría",
      "50": "Autoconciencia Emocional",
      "30": "Autocontrol Emocional",
      "20": "Motivación al Logro"
    },
    {
      "nombre": "amor",
      "50": "Desarrollo de la Relaciones",
      "30": "Desarrollar y Estimular a los demás",
      "20": "Autoconciencia Emocional"
    },
    {
      "nombre": "ansiedad",
      "50": "Adaptabilidad",
      "30": "Autocontrol Emocional",
      "20": "Tolerancia a la Frustración"
    },
    {
      "nombre": "apoyo",
      "50": "Colaboración y Cooperación",
      "30": "Influencia",
      "20": "Liderazgo"
    },
    {
      "nombre": "arrepentimiento",
      "50": "Tolerancia a la Frustración",
      "30": "Autoconciencia Emocional",
      "20": "Adaptabilidad"
    },
    {
      "nombre": "asombro",
      "50": "Adaptabilidad",
      "30": "Autocontrol Emocional",
      "20": "Autoconciencia Emocional"
    },
    {
      "nombre": "cansancio",
      "50": "Autoconciencia Emocional",
      "30": "Adaptabilidad",
      "20": "Motivación al Logro"
    },
    {
      "nombre": "confusion",
      "50": "Autoconciencia Emocional",
      "30": "Asertividad",
      "20": "Adaptabilidad"
    },
    {
      "nombre": "culpa",
      "50": "Influencia",
      "30": "Empatía",
      "20": "Comprensión Organizativa"
    },
    {
      "nombre": "curiosidad",
      "50": "Desarrollar y Estimular a los demás",
      "30": "Conciencia Crítica",
      "20": "Comprensión Organizativa"
    },
    {
      "nombre": "decepcion",
      "50": "Desarrollo de la Relaciones",
      "30": "Desarrollar y Estimular a los demás",
      "20": "Percepción y Comprensión Emocional"
    },
    {
      "nombre": "desden",
      "50": "Desarrollo de la Relaciones",
      "30": "Colaboración y Cooperación",
      "20": "Manejo de Conflictos"
    },
    {
      "nombre": "deseo",
      "50": "Motivación al Logro",
      "30": "Autocontrol Emocional",
      "20": "Autoconciencia Emocional"
    },
    {
      "nombre": "desesperacion",
      "50": "Autocontrol Emocional",
      "30": "Tolerancia a la Frustración",
      "20": "Asertividad"
    },
    {
      "nombre": "dolor",
      "50": "Autocontrol Emocional",
      "30": "Autoestima",
      "20": "Tolerancia a la Frustración"
    },
    {
      "nombre": "duda",
      "50": "Motivación al Logro",
      "30": "Asertividad",
      "20": "Adaptabilidad"
    },
    {
      "nombre": "frustacion",
      "50": "Tolerancia a la Frustración",
      "30": "Adaptabilidad",
      "20": "Autoestima"
    },
    {
      "nombre": "honestidad",
      "50": "Comunicación Asertiva",
      "30": "Colaboración y Cooperación",
      "20": "Liderazgo"
    },
    {
      "nombre": "impresion",
      "50": "Autoconciencia Emocional",
      "30": "Comprensión Organizativa",
      "20": "Percepción y Comprensión Emocional"
    },
    {
      "nombre": "indignacion",
      "50": "Manejo de Conflictos",
      "30": "Liderazgo",
      "20": "Percepción y Comprensión Emocional"
    },
    {
      "nombre": "interes",
      "50": "Percepción y Comprensión Emocional",
      "30": "Influencia",
      "20": "Liderazgo"
    },
    {
      "nombre": "molestia",
      "50": "Desarrollar y Estimular a los demás",
      "30": "Autoconciencia Emocional",
      "20": "Desarrollo de la Relaciones"
    },
    {
      "nombre": "odio",
      "50": "Desarrollar y Estimular a los demás",
      "30": "Empatía",
      "20": "Autocontrol Emocional"
    },
    {
      "nombre": "orgullo",
      "50": "Autoestima",
      "30": "Liderazgo",
      "20": "Influencia"
    },
    {
      "nombre": "preocupacion",
      "50": "Empatía",
      "30": "Percepción y Comprensión Emocional",
      "20": "Desarrollar y Estimular a los demás"
    },
    {
      "nombre": "remordimiento",
      "50": "Tolerancia a la Frustración",
      "30": "Manejo de Conflictos",
      "20": "Adaptabilidad"
    },
    {
      "nombre": "sorpresa",
      "50": "Desarrollar y Estimular a los demás",
      "30": "Adaptabilidad",
      "20": "Percepción y Comprensión Emocional"
    },
    {
      "nombre": "tranquilidad",
      "50": "Autocontrol Emocional",
      "30": "Tolerancia a la Frustración",
      "20": "Adaptabilidad"
    },
    {
      "nombre": "tristeza",
      "50": "Autoconciencia Emocional",
      "30": "Autocontrol Emocional",
      "20": "Tolerancia a la Frustración"
    },
    {
      "nombre": "verguenza",
      "50": "Autoconciencia Emocional",
      "30": "Autoestima",
      "20": "Autocontrol Emocional"
    }
    
]
def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    @app.route('/')
    def hello():
        return render_template('index.html')
    
    

    # endpoint to run the script
    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        if request.method == 'POST':
            user_input = request.form['user_input']
            # Aquí llamamos a la función que procesa el input
            prediction, user_text = run_inference(user_input)
            return render_template('/predict.html', prediction=prediction, user_text=user_text)

        return render_template('/predict.html')
    
    @app.route('/predict2', methods=['GET', 'POST'])
    def predict2():
        if request.method == 'POST':
            user_input = request.form['user_input']
            # Aquí llamamos a la función que procesa el input
            prediction, user_text, factor_50, factor_30, factor_20 = run_inference2(user_input)
            return render_template('/predict2.html', prediction=prediction, user_text=user_text, factor_50=factor_50, factor_30=factor_30, factor_20=factor_20)

        return render_template('/predict2.html')

    return app

def run_inference(text):
    # Verificar si CUDA (GPU) está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ruta al archivo de dataset
    dataset_path = 'dataset_limpio.xlsx'

    # Cargar el dataset
    df = pd.read_excel(dataset_path)

    # Asegurarse de que la columna 'Emoción' sea categórica
    df['Emoción'] = df['Emoción'].astype('category')

    # Obtener las categorías de las emociones
    emotion_categories = df['Emoción'].cat.categories

    # Cargar el modelo y el tokenizador guardados en GPU
    model = RobertaForSequenceClassification.from_pretrained('./saved_models').to(device)
    tokenizer = RobertaTokenizer.from_pretrained('./saved_models')

    # Tokenizar el texto de prueba
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    # Hacer predicción
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).item()

    # Decodificar la predicción
    emotion = emotion_categories[predictions]
    return emotion, text

def run_inference2(text):
    # Verificar si CUDA (GPU) está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ruta al archivo de dataset
    dataset_path = 'dataset final.xlsx'

    # Cargar el dataset
    df = pd.read_excel(dataset_path)

    # Asegurarse de que la columna 'Emoción' sea categórica
    df['Emoción'] = df['Emoción'].astype('category')

    # Obtener las categorías de las emociones
    emotion_categories = df['Emoción'].cat.categories

    # Cargar el modelo y el tokenizador guardados en GPU
    model = RobertaForSequenceClassification.from_pretrained('./new_model').to(device)
    tokenizer = RobertaTokenizer.from_pretrained('./new_model')

    # Tokenizar el texto de prueba
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    # Hacer predicción
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).item()

    # Decodificar la predicción
    emotion = emotion_categories[predictions]

    # Buscar la emoción en el JSON
    emotion_data = next((item for item in emotions if item["nombre"] == emotion), None)
    
    if emotion_data:
        factor_50 = emotion_data["50"]
        factor_30 = emotion_data["30"]
        factor_20 = emotion_data["20"]
    else:
        factor_50 = factor_30 = factor_20 = "No disponible"

    return emotion, text, factor_50, factor_30, factor_20


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
