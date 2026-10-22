import numpy as np
import tensorflow as tf

class ClassifierService:
    def __init__(self):
        self.model = None
        self.classes = [] # Adicione a lista de suas classes aqui se necessário

    def load_model(self, model_path: str):
        """Carrega o modelo do TensorFlow a partir do caminho especificado"""
        # Carrega o modelo de forma compatível
        self.model = tf.keras.models.load_model(model_path)
        # Exemplo padrão de classes do CIFAR-10 (ajuste se o seu for diferente)
        self.classes = ['avião', 'automóvel', 'pássaro', 'gato', 'cervo', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

    def predict(self, pixels_list: list) -> tuple:
        """Executa a inferência a partir dos dados estruturados em memória"""
        if self.model is None:
            raise RuntimeError("Modelo não inicializado na memória RAM.")

        # Reconstrói o formato original da imagem (1, 32, 32, 3) requisitado pela CNN
        image_array = np.array(pixels_list, dtype=np.float32).reshape(1, 32, 32, 3)

        predictions = self.model.predict(image_array, verbose=0)
        score = tf.nn.softmax(predictions)
        id_classe = np.argmax(score)
        confianca = float(np.max(score)) * 100

        return self.classes[id_classe], f"{confianca:.2f}%"

# Criando a instância necessária que o main.py tenta importar
classifier_service = ClassifierService()

