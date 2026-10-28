
# ImageClass API

API REST desenvolvida com FastAPI e TensorFlow/Keras para classificar imagens utilizando uma rede neural convolucional (CNN) treinada com o conjunto de dados CIFAR-10.

O projeto demonstra a integração de um modelo de Machine Learning a uma API, com carregamento do modelo na inicialização, processamento de imagens e disponibilização de endpoints HTTP.

## Funcionalidades

- Classificação de imagens por meio de uma CNN.
- Processamento e redimensionamento de imagens para 32 × 32 pixels.
- Normalização dos valores dos pixels.
- Carregamento do modelo durante a inicialização da API.
- Endpoint de verificação de saúde do serviço.
- Documentação interativa com Swagger UI.

## Tecnologias utilizadas

- Python
- FastAPI
- Uvicorn
- TensorFlow / Keras
- NumPy
- Pillow
- Pydantic

## Estrutura do projeto

```text
ImageClass_API/
├── app/
│   ├── routers/
│   │   └── predict.py
│   ├── main.py
│   ├── schemas.py
│   └── services.py
├── data/
├── models/
│   └── modelo_cifar10.keras
├── notebooks/
├── .gitignore
├── requirements.txt
├── treinar_modelo.py
├── explorar_cifar10.py
└── salvar_modelo.py
```

O arquivo `modelo_cifar10.keras` contém o modelo treinado utilizado pela API. A pasta `venv/`, quando criada, é destinada ao ambiente virtual local e não deve ser enviada ao GitHub.

## Requisitos

- Python instalado e compatível com as versões das dependências do projeto.
- Git, para clonar o repositório.
- Arquivo do modelo disponível em `models/modelo_cifar10.keras`.

## Instalação

Clone o repositório:

```bash
git clone https://github.com/Dora-bit767/ImageClass_API.git
cd ImageClass_API
```

Crie um ambiente virtual:

```powershell
python -m venv venv
```

Ative o ambiente virtual no Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install python-multipart
```

## Executar a API

Na raiz do projeto, com o ambiente virtual ativado, execute:

```powershell
python -m uvicorn app.main:app --reload
```

A API ficará disponível localmente em:

- **Documentação Swagger:** http://127.0.0.1:8000/docs
- **Verificação de saúde:** http://127.0.0.1:8000/health

Para encerrar o servidor, pressione `Ctrl + C` no terminal.

## Endpoints

### GET `/health`

Verifica se o modelo está carregado na memória.

Exemplo de resposta com HTTP 200:

```json
{
  "status": "Healthy",
  "servico": "Ativo",
  "modelo_em_ram": true
}
```

### POST `/predict`

Recebe uma imagem por meio de `multipart/form-data`, processa o arquivo e retorna a classe prevista e a confiança calculada.

Para testar:

1. Abra http://127.0.0.1:8000/docs.
2. Expanda `POST /predict`.
3. Clique em **Try it out**.
4. Selecione uma imagem no campo `file`.
5. Clique em **Execute**.

Exemplo ilustrativo de formato de resposta (valores fictícios; a classe e a confiança variam conforme a imagem enviada):

```json
{
  "classe_detectada": "sapo",
  "confianca": "87.42%"
}
```

Os valores retornados dependem da imagem enviada e das previsões do modelo.

## Classes do CIFAR-10

O modelo trabalha com dez categorias:

1. Avião
2. Automóvel
3. Pássaro
4. Gato
5. Veado
6. Cachorro
7. Sapo
8. Cavalo
9. Navio
10. Caminhão

## Avaliação do modelo

O modelo foi avaliado utilizando o conjunto oficial de teste do CIFAR-10, composto por 10.000 imagens.


Resultados obtidos com o modelo salvo em `models/modelo_cifar10.keras`:

- **Acurácia (accuracy):** 67,96%
- **Perda (loss):** 0,9406

A avaliação foi realizada por meio do script `notebooks/avaliar_modelo.py`, sem executar um novo treinamento.

Os resultados representam o desempenho do modelo nesse conjunto de teste e não garantem a mesma taxa de acerto em imagens externas.

## Treinamento do modelo

O script `treinar_modelo.py` realiza o treinamento, a avaliação e o salvamento do modelo na pasta `models/`.


**Atenção:** executar esse script novamente treina o modelo outra vez e pode substituir o arquivo salvo. Para utilizar a API, não é necessário treinar novamente se o arquivo do modelo já estiver disponível.

## Status do projeto

- API iniciada e testada localmente.
- Endpoint `GET /health` validado com HTTP 200.
- Endpoint `POST /predict` testado com classificação de imagem.
- Documentação interativa disponível pelo Swagger UI.

## Autoria

Projeto desenvolvido para demonstrar conhecimentos em Python, análise de dados, Machine Learning e desenvolvimento de APIs.

## Vídeo de apresentação

[Assistir ao vídeo do projeto no Google Drive](COLE_AQUI_O_LINK_DO_VIDEO)

