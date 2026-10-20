# 1. Importar as bibliotecas necessárias
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# 2. Definir os nomes das 10 categorias
classes = [
    "avião", "automóvel", "pássaro", "gato", "veado",
    "cachorro", "sapo", "cavalo", "navio", "caminhão"
]

# 3. Carregar os dados do CIFAR-10
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)

# 4. Conferir o tamanho dos conjuntos
print("Imagens de treinamento:", x_train.shape)
print("Rótulos de treinamento:", y_train.shape)
print("Imagens de teste:", x_test.shape)
print("Rótulos de teste:", y_test.shape)

# 5. Criar e salvar a primeira imagem do treinamento
indice = 1
plt.figure(figsize=(3, 3)) # Define um tamanho adequado para a imagem 32x32
plt.imshow(x_train[indice])
nome_classe = classes[int(y_train[indice][0])]
plt.title(f"Categoria: {nome_classe}")
plt.axis("off")

# Em vez de plt.show(), salvamos o arquivo na pasta do projeto
nome_arquivo = "exemplo_cifar10.png"
plt.savefig(nome_arquivo, bbox_inches='tight')
print(f"\n[SUCESSO] Imagem salva com sucesso como '{nome_arquivo}'!")

# Força o Windows a abrir a imagem borrada padrão na tela
os.system(f"start {nome_arquivo}")

