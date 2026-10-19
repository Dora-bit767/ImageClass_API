from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers.predict import router as predict_router
from app.services import classifier_service

# Gerenciamento Otimizado do Ciclo de Vida (Fase 3 - Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Executa estritamente na inicialização (Startup) do servidor
    MODEL_PATH = ".ImageClass-API/models/modelo_cifar10.keras"
    print(f"[LIFESPAN] Carregando modelo binário de: {MODEL_PATH} para a memória RAM uma única vez...")
    
    classifier_service.load_model(MODEL_PATH)
    yield
    
    # Executa no encerramento (Shutdown)
    print("[LIFESPAN] Encerrando serviços da API.")

# Inicialização da aplicação com o Lifespan configurado
app = FastAPI(
    title="ImageClass API", 
    description="API preditiva modular de alta performance para o CIFAR-10",
    lifespan=lifespan
)

# Inclusão das rotas da pasta routers (Fase 2)
app.include_router(predict_router)

