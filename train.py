import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopw = stopwords.words('spanish')
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import RobertaTokenizer

# Cargar el archivo de Excel
file_path = 'dataset.xlsx'  # Asegúrate de cambiar la ruta al archivo correcto
df = pd.read_excel(file_path)

# Seleccionar solo 1000 frases
df = df.sample(n=18000, random_state=42)

# Visualizar las primeras filas del dataframe
print(df.head())

# Seleccionar columnas relevantes
df = df[['Frase', 'Emoción']]

# Codificar las emociones
df['Emoción'] = df['Emoción'].astype('category')
df['encoded_emotion'] = df['Emoción'].cat.codes

# Crear listas de textos y etiquetas
data_texts = df['Frase'].to_list()
data_labels = df['encoded_emotion'].to_list()

# División de los datos en entrenamiento, validación y prueba
train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=0.2, random_state=0)
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.01, random_state=0)

from transformers import RobertaTokenizer

# Tokenizador
tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')

# Tokenizar datos de entrenamiento y validación
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Crear datasets
import torch

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

# Definir el modelo
model = RobertaForSequenceClassification.from_pretrained('PlanTL-GOB-ES/roberta-base-bne', num_labels=len(df['encoded_emotion'].unique()))

# Configurar los parámetros de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Crear el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Entrenar el modelo
trainer.train()

# Evaluar el modelo
trainer.evaluate()

# Guardar el modelo
model.save_pretrained('./modelcpu')
tokenizer.save_pretrained('./modelcpu')
