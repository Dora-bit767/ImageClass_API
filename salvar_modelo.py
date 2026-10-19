import tensorflow as tf

# 1. Carregar os dados novamente apenas para fazer a avaliação oficial
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test / 255.0

# 2. Recriar a estrutura idêntica para receber os pesos treinados (ou recarregar o histórico)
# Para simplificar e salvar diretamente o modelo treinado em um arquivo físico:
print("[INFO] Carregando a estrutura para salvamento...")

# Como o modelo anterior ficou na memória RAM do outro script, vamos ajustar o próprio 'treinar_modelo.py' 
# ou salvar diretamente adicionando uma linha nele. 
# --- CÓDIGO DA ETAPA 4: AVALIAR E SALVAR O MODELO ---
print("\n[INFO] Avaliando o modelo final com os dados de teste...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Acurácia final nos dados de teste: {test_acc*100:.2f}%")

# Salvar o modelo treinado em um arquivo compactado
nome_modelo = "modelo_cifar10.keras"
model.save(nome_modelo)
print(f"[SUCESSO] Modelo salvo fisicamente como '{nome_modelo}'!")
