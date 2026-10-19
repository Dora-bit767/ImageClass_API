import numpy as np
import tensorflow as tf

class ImageClassificationService:
    def __init__(self):
        self.model = None
        self.classes = [
            "avião", "automóvel", "pássaro", "gato", "veado",
            "cachorro", "sapo", "cavalo", "navio", "caminhão"
        ]

    def load_model(self, model_path: str):
        """Carrega o artefato binário na memória RAM uma única vez"""
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, pixels_list: list) -> tuple:
        """Executa a inferência a partir dos dados estruturados em memória"""
        if self.model is None:
            raise RuntimeError("Modelo não inicializado na memória RAM.")
        
        # Reconstrói o formato original da imagem (1, 32, 32, 3) requisitado pela CNN
        image_array = np.array(pixels_list).reshape(1, 32, 32, 3)
        
        predictions = self.model.predict(image_array)
        score = tf.nn.softmax(predictions)
        
        id_classe = np.argmax(score)
        confianca = float(np.max(score)) * 100
        
        return self.classes[id_classe], f"{confianca:.2f}%"

# Instância única global do serviço para reuso nas rotas
classifier_service = ImageClassificationService()
