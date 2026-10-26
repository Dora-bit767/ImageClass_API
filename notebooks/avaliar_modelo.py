import tensorflow as tf
from pathlib import Path

# Localizar o modelo já salvo
BASE_DIR = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = BASE_DIR / "models" / "modelo_cifar10.keras"

# Carregar o modelo existente, sem treinar novamente
model = tf.keras.models.load_model(CAMINHO_MODELO)

# Carregar os dados oficiais de teste do CIFAR-10
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test / 255.0

# Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f"\nAcurácia nos dados de teste: {test_acc * 100:.2f}%")
print(f"Perda nos dados de teste: {test_loss:.4f}")