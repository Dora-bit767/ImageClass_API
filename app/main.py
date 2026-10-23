from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from app.routers.predict import router as predict_router
from app.services import classifier_service


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "modelo_cifar10.keras"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[LIFESPAN] Carregando modelo de: {MODEL_PATH}")
    classifier_service.load_model(str(MODEL_PATH))

    yield

    print("[LIFESPAN] Encerrando serviços da API.")


app = FastAPI(
    title="ImageClass API",
    description="API preditiva modular para classificação de imagens CIFAR-10",
    lifespan=lifespan,
)

app.include_router(predict_router)