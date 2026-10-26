import io
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from app.schemas import PredictionOutputSchema
from app.services import classifier_service

router = APIRouter()

# Endpoint 1: Monitoramento de Saúde (Fase 5 - GET /health)
@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if classifier_service.model is not None:
        return {
            "status": "Healthy",
            "servico": "Ativo",
            "modelo_em_ram": True
        }
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Modelo de ML ausente da memória RAM."
    )

# Endpoint 2: Inferência Sob Demanda (Fase 5 - POST /predict)
@router.post("/predict", response_model=PredictionOutputSchema, status_code=status.HTTP_200_OK)
async def predict_image(file: UploadFile = File(...)):
    # Validação estrita do formato do arquivo recebido (Fase 4)
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="O arquivo enviado precisa ser uma imagem válida."
        )
    
    try:
        # Lendo os bytes da imagem enviada
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Pré-processamento: Redimensionamento e Normalização (Fase 1 e 4)
        image = image.resize((32, 32))
        pixels_array = np.array(image) / 255.0
        pixels_list = pixels_array.tolist()  # Converte em lista para o serviço matemático
        
        # Invoca a camada de serviço desacoplada (Fase 2)
        classe, confianca = classifier_service.predict(pixels_list)
        return {"classe_detectada": classe, "confianca": confianca}
        
    except HTTPException:
        raise

    except (Image.UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não foi possível processar o arquivo enviado. Verifique se é uma imagem válida."
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno no processamento da inferência."
        ) from e
