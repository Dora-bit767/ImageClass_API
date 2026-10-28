import asyncio

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.services import classifier_service
from app.routers.predict import health_check


def test_health_com_modelo_carregado():
    modelo_original = classifier_service.model

    try:
        classifier_service.model = object()

        resultado = asyncio.run(health_check())

        assert resultado["status"] == "Healthy"
        assert resultado["servico"] == "Ativo"
        assert resultado["modelo_em_ram"] is True

    finally:
        classifier_service.model = modelo_original


def test_health_sem_modelo():
    modelo_original = classifier_service.model

    try:
        classifier_service.model = None

        try:
            asyncio.run(health_check())
            assert False, "Era esperado um erro HTTP 500"
        except HTTPException as exc:
            assert exc.status_code == 500
            assert "Modelo de ML ausente" in exc.detail

    finally:
        classifier_service.model = modelo_original


def test_predict_rejeita_arquivo_que_nao_e_imagem():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "arquivo.txt",
                    b"conteudo de teste",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 422