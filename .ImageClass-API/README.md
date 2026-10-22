# ImageClass API 🚀

Uma API RESTful assíncrona, robusta e modular desenvolvida com **FastAPI** e **TensorFlow/Keras**, projetada para a classificação inteligente de imagens utilizando uma Rede Neural Convolucional (CNN) treinada sobre o dataset **CIFAR-10**.

---

## 📂 Estrutura Modular da Aplicação (Fase 2)

O projeto foi arquitetado seguindo as melhores práticas de engenharia de software e MLOps, garantindo o desacoplamento de responsabilidades através da seguinte estrutura em camadas:

```text
ImageClass_API/
├── .ImageClass-API/
│   ├── app/
│   │   ├── routers/
│   │   │   └── predict.py   # Implementação dos endpoints e controle HTTP
│   │   ├── main.py          # Ponto de entrada e gerenciamento do Lifespan
│   │   ├── schemas.py       # Modelagem e validação estrita com Pydantic
│   │   └── services.py      # Camada isolada de inferência matemática do modelo
│   └── models/
│       └── modelo_cifar10.keras  # Artefato serializado da Inteligência Artificial
├── venv/                    # Ambiente virtual isolado do Python
├── explorar_cifar10.py      # Pipeline offline de análise de dados original
├── treinar_modelo.py        # Script automatizado de treinamento e exportação da CNN
└── requirements.txt         # Gerenciamento de dependências e reprodutibilidade
```

---

## 🛠️ Tecnologias e Frameworks Utilizados

* **Python 3.14+**
* **TensorFlow / Keras 2.x:** Construção, processamento e inferência da rede neural convolucional.
* **FastAPI:** Framework web moderno de alta performance baseado em tipos estáticos.
* **Pydantic:** Validação estrita de contratos de dados em tempo de execução.
* **Uvicorn:** Servidor ASGI para execução assíncrona e baixa latência.
* **Pillow & NumPy:** Pipelines eficientes de processamento de matrizes de pixels in-memory.

---

## 💡 Otimizações Técnicas Implementadas

* **Gerenciamento de Ciclo de Vida (Lifespan):** O arquivo `.keras` é injetado diretamente na memória RAM uma única vez durante a rotina de inicialização (`startup`) do servidor usando `@asynccontextmanager`. É terminantemente proibida a leitura repetitiva de disco por requisição, mitigando gargalos de I/O.
* **Tratamento de Exceções Semântico:** Interceptação automática de arquivos corrompidos ou payloads inválidos, retornando os status codes HTTP corretos (`200 OK`, `422 Unprocessable Entity` ou `500 Internal Server Error`) impedindo quedas (*crashes*) inesperadas na aplicação.

---

## 🚀 Guia de Execução Local

### 1. Clonar o Repositório e Configurar Ambiente
```bash
git clone https://github.com
cd ImageClass_API
```

### 2. Ativar o Ambiente Virtual (Windows)
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Instalar Dependências Obrigatórias
```bash
pip install -r requirements.txt
pip install python-multipart
```

### 4. Inicializar a API FastAPI
```bash
python -m uvicorn app.main:app --app-dir .ImageClass-API --reload
```

---

## 📊 Documentação dos Endpoints (Swagger)

A documentação interativa automatizada pode ser acessada em tempo real através do navegador assim que o servidor local estiver ativo.

👉 **Acesse a interface gráfica para testes em:** **[http://127.0.0](http://127.0.0)**

### 🛠️ Alternativa por Linha de Comando (cURL)

Se preferir validar o funcionamento do sistema diretamente pelo seu terminal, você pode disparar as seguintes requisições:

#### 1. Check de Saúde Geral
* **Rota:** `GET /health`
* **Comando:**
```bash
curl -X 'GET' 'http://127.0.0' -H 'accept: application/json'
```
* **Resposta Esperada (HTTP 200):**
```json
{
  "status": "Healthy",
  "servico": "Ativo",
  "modelo_em_ram": true
}
```

#### 2. Inferência Preditiva Sob Demanda
* **Rota:** `POST /predict`
* **Tipo de Entrada:** `multipart/form-data`
* **Comando:** *(Substitua `caminho/foto.jpg` pelo caminho real de uma imagem no seu computador)*
```bash
curl -X 'POST' \
  'http://127.0.0' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@caminho/foto.jpg;type=image/jpeg'
```
* **Contrato de Saída Validado (HTTP 200):**
```json
{
  "classe_detectada": "sapo",
  "confianca": "87.42%"
}
```

---

## 📹 Link das Evidências (Entrega Acadêmica)
* **Gravação de Vídeo de Apresentação:** [Insira aqui o link do seu Google Drive em modo leitor]
