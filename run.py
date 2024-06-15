from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
import pandas as pd 

print(torch.cuda.is_available())
# Verificar si CUDA (GPU) está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

# Ejemplo de texto para inferencia
test_text = "Conchetumare"

# Tokenizar el texto de prueba
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True).to(device)

# Hacer predicción
with torch.no_grad():
    outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1).item()

# Decodificar la predicción
emotion = emotion_categories[predictions]
print(emotion)
