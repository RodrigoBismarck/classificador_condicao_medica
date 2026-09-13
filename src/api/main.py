"""
API FastAPI para classificação de condições médicas.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.model.predict import MotorPredição

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métricas Prometheus
contador_requisicoes = Counter(
    "requisicoes_total", "Total de requisições à API", ["metodo", "endpoint", "status"]
)

tempos_resposta = Histogram(
    "tempo_resposta_segundos", "Tempo de resposta em segundos", ["endpoint"]
)

contador_predicoes = Counter("predicoes_total", "Total de predições realizado", ["classe"])


# Modelos Pydantic
class RequisicaoPredição(BaseModel):
    """Modelo para requisição de predição única."""

    texto: str = Field(..., min_length=1, max_length=5000, description="Texto médico a classificar")

    class Config:
        example = {"texto": "Advanced pancreatic cancer with metastasis"}


class RequisicaoPredçãoLote(BaseModel):
    """Modelo para requisição de predição em lote."""

    textos: List[str] = Field(
        ..., min_items=1, max_items=100, description="Lista de textos médicos"
    )

    class Config:
        example = {
            "textos": [
                "Advanced pancreatic cancer with metastasis",
                "Atrial fibrillation detected on ECG with rapid ventricular response",
                "Fever, chills, and signs of sepsis of unknown origin",
            ]
        }


class RespostaPredição(BaseModel):
    """Modelo para resposta de predição."""

    texto_original: str
    classe_predita: str
    id_classe: int
    confianca: float
    todas_probabilidades: Dict[str, float]


class RespostaPredçãoLote(BaseModel):
    """Modelo para resposta de predição em lote."""

    total: int
    predicoes: List[RespostaPredição]


class RespostaVitals(BaseModel):
    """Modelo para informações de vitalidade."""

    status: str
    timestamp: str
    versao_modelo: str


class InfoModelo(BaseModel):
    """Modelo para informações do modelo."""

    nome: str
    tipo: str
    versao: str
    n_features: int
    n_classes: int
    classes: List[str]
    arquitetura: Dict


# Instância global do motor de predição
motor_predicao: Optional[MotorPredição] = None


@asynccontextmanager
async def ciclo_vida(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação (startup e shutdown).
    """
    # Startup
    global motor_predicao
    try:
        caminho_modelo = os.getenv("MODEL_PATH", "data/models/modelo_medico.pkl")
        caminho_vetorizador = os.getenv("VECTORIZER_PATH", "data/models/vetorizador_medico.pkl")

        if not os.path.exists(caminho_modelo) or not os.path.exists(caminho_vetorizador):
            logger.warning("Arquivos de modelo não encontrados. Caminhos esperados:")
            logger.warning(f"  - Modelo: {caminho_modelo}")
            logger.warning(f"  - Vetorizador: {caminho_vetorizador}")
            logger.warning("Use 'python train_model.py' para treinar um novo modelo")
        else:
            motor_predicao = MotorPredição(caminho_modelo, caminho_vetorizador)
            logger.info("Motor de predição iniciado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao iniciar motor de predição: {e}")

    yield

    # Shutdown
    logger.info("Encerrando aplicação")


# Criar app
app = FastAPI(
    title="API de Classificação de Condições Médicas",
    description="API para classificação de condições médicas usando Machine Learning",
    version="1.0.0",
    lifespan=ciclo_vida,
)


# Middleware para logging e métricas
@app.middleware("http")
async def middleware_metricas(requisicao, chamar_proxima):
    """Middleware para registrar métricas de requisições."""
    from time import time

    inicio = time()
    resposta = await chamar_proxima(requisicao)
    duracao = time() - inicio

    endpoint = requisicao.url.path
    tempos_resposta.labels(endpoint=endpoint).observe(duracao)
    contador_requisicoes.labels(
        metodo=requisicao.method, endpoint=endpoint, status=resposta.status_code
    ).inc()

    return resposta


# Endpoints


@app.get(
    "/health",
    response_model=RespostaVitals,
    summary="Verificar saúde da API",
    tags=["Sistema"],
)
async def verificar_saude():
    """
    Verifica se a API está operacional.

    Returns:
        Dicionário com status da API
    """
    return {
        "status": "ok" if motor_predicao else "model_not_loaded",
        "timestamp": datetime.now().isoformat(),
        "versao_modelo": "1.0.0",
    }


@app.post(
    "/predict",
    response_model=RespostaPredição,
    summary="Predição única",
    tags=["Predições"],
)
async def prever(requisicao: RequisicaoPredição):
    """
    Realiza predição para um único texto médico.

    Args:
        requisicao: Objeto contendo o texto a classificar

    Returns:
        Predição com classe, confiança e probabilidades

    Raises:
        HTTPException: Se o modelo não estiver carregado
    """
    if not motor_predicao:
        raise HTTPException(status_code=503, detail="Modelo não está carregado. Tente mais tarde.")

    try:
        resultado = motor_predicao.prever(requisicao.texto)
        contador_predicoes.labels(classe=resultado["classe_predita"]).inc()
        return resultado
    except Exception as e:
        logger.error(f"Erro durante predição: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")


@app.post(
    "/predict-batch",
    response_model=RespostaPredçãoLote,
    summary="Predição em lote",
    tags=["Predições"],
)
async def prever_lote(requisicao: RequisicaoPredçãoLote):
    """
    Realiza predição para múltiplos textos médicos.

    Args:
        requisicao: Objeto contendo lista de textos

    Returns:
        Lista de predições

    Raises:
        HTTPException: Se o modelo não estiver carregado
    """
    if not motor_predicao:
        raise HTTPException(status_code=503, detail="Modelo não está carregado. Tente mais tarde.")

    try:
        resultado = motor_predicao.prever_lote(requisicao.textos)

        # Registra predições em métricas
        for pred in resultado["predicoes"]:
            contador_predicoes.labels(classe=pred["classe_predita"]).inc()

        return resultado
    except Exception as e:
        logger.error(f"Erro durante predição em lote: {e}")
        raise HTTPException(status_code=500, detail="Erro interno do servidor")


@app.get(
    "/model-info",
    response_model=InfoModelo,
    summary="Informações do modelo",
    tags=["Modelo"],
)
async def obter_info_modelo():
    """
    Retorna informações detalhadas sobre o modelo.

    Returns:
        Informações do modelo (arquitetura, classes, etc)

    Raises:
        HTTPException: Se o modelo não estiver carregado
    """
    if not motor_predicao:
        raise HTTPException(status_code=503, detail="Modelo não está carregado")

    return motor_predicao.obter_info_modelo()


@app.get("/metrics", summary="Métricas Prometheus", tags=["Monitoramento"])
async def obter_metricas():
    """
    Expõe métricas no formato Prometheus.

    Returns:
        Métricas em formato text/plain
    """
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    porta = int(os.getenv("API_PORT", 8000))

    uvicorn.run("src.api.main:app", host=host, port=porta, reload=True, log_level="info")
