import numpy as np
import tensorflow as tf


class ImageClassificationService:
    def __init__(self):
        self.model = None
        self.classes = [
            "avião",
            "automóvel",
            "pássaro",
            "gato",
            "veado",
            "cachorro",
            "sapo",
            "cavalo",
            "navio",
            "caminhão",
        ]

    def load_model(self, model_path: str):
        """Carrega o modelo na memória uma única vez."""
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, pixels_list: list) -> tuple:
        """Realiza a previsão a partir dos pixels da imagem."""
        if self.model is None:
            raise RuntimeError("O modelo não foi carregado na memória.")

        # Reconstrói o formato esperado pela CNN: (1, 32, 32, 3).
        image_array = np.array(pixels_list).reshape(1, 32, 32, 3)

        predictions = self.model.predict(image_array, verbose=0)
        score = tf.nn.softmax(predictions)

        id_classe = int(np.argmax(score))
        confianca = float(np.max(score)) * 100

        return self.classes[id_classe], f"{confianca:.2f}%"


# Instância única do serviço, compartilhada pelas rotas.
classifier_service = ImageClassificationService()