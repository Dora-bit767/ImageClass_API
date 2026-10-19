import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Carregar e normalizar os dados (mudar pixels de 0-255 para 0-1)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Construir a estrutura da Rede Neural Convolucional (CNN)
model = models.Sequential([
    # Camadas Convolucionais (enxergam padrões na imagem como olhos, patas, rodas)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Camadas Densas (tomam a decisão final de qual é a categoria)
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10) # 10 saídas correspondentes às 10 classes do CIFAR-10
])

# 3. Compilar o modelo
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 4. Exibir o resumo da estrutura no terminal
model.summary()

# 5. Iniciar o treinamento (apenas 5 épocas para ser rápido e seguro)
print("\n[INFO] Iniciando o treinamento do modelo...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("[SUCESSO] Treinamento concluído!")

# --- ADICIONE ESTE BLOCO ABAIXO PARA SALVAR NO LOCAL CORRETO ---
print("\n[INFO] Avaliando o modelo final com os dados de teste...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Acurácia final nos dados de teste: {test_acc*100:.2f}%")

# Salvar o modelo treinado dentro da pasta .ImageClass-API/models/
nome_modelo = ".ImageClass-API/models/modelo_cifar10.keras"
model.save(nome_modelo)
print(f"[SUCESSO] Modelo salvo fisicamente em '{nome_modelo}'!")


