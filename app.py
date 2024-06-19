import os
from flask import Flask, request, render_template
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
import pandas as pd

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
            prediction, user_text = run_inference2(user_input)
            return render_template('/predict2.html', prediction=prediction, user_text=user_text)

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
    return emotion, text

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
