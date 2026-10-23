from pydantic import BaseModel, Field
from typing import List

# Validação do contrato de entrada de dados estruturados (matriz de pixels normalizada)
class ImageInputSchema(BaseModel):
    # Uma imagem 32x32 com 3 canais RGB convertida em lista plana de floats para validação do Pydantic
    pixels: List[float] = Field(
        ..., 
        description="Lista linear contendo os 3072 valores de pixels (32x32x3) normalizados."
    )

    # Exemplo de regra estrita exigida pelo critério de avaliação da Fase 4
    @classmethod
    def validate_pixels_range(cls, v):
        for pixel in v:
            if pixel < 0.0 or pixel > 1.0:
                raise ValueError("Cada pixel deve estar normalizado entre 0.0 e 1.0 (restrição ge=0.0 e le=1.0)")
        return v

# Contrato de saída estruturado
class PredictionOutputSchema(BaseModel):
    classe_detectada: str = Field(..., description="Nome da categoria prevista pelo modelo")
    confianca: str = Field(..., description="Porcentagem de confiança do acerto")
