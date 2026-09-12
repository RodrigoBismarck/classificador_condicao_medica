"""
Script para exportar o modelo treinado para formato ONNX.
"""

import logging
import os
from pathlib import Path

try:
    from src.model.optimize import OtimizadorONNX
except ImportError:
    print("Erro: ONNX não está instalado. Install com: pip install skl2onnx onnxruntime")
    exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Exporta modelo Random Forest + TF-IDF para formato ONNX.
    """
    logger.info("=" * 50)
    logger.info("EXPORTANDO MODELO PARA ONNX")
    logger.info("=" * 50)

    try:
        # Caminhos dos arquivos
        caminho_modelo = "data/models/modelo_medico.pkl"
        caminho_vetorizador = "data/models/vetorizador_medico.pkl"
        caminho_onnx = "data/models/modelo_medico.onnx"

        # Verifica se arquivos existem
        if not os.path.exists(caminho_modelo):
            logger.error(f"\nErro: Modelo não encontrado em {caminho_modelo}")
            logger.error("Execute 'python train_model.py' para treinar um modelo")
            return False

        if not os.path.exists(caminho_vetorizador):
            logger.error(f"\nErro: Vetorizador não encontrado em {caminho_vetorizador}")
            logger.error("Execute 'python train_model.py' para treinar um modelo")
            return False

        logger.info(f"\n1. Carregando arquivos...")
        logger.info(f"   Modelo: {caminho_modelo}")
        logger.info(f"   Vetorizador: {caminho_vetorizador}")

        # Cria otimizador
        otimizador = OtimizadorONNX(caminho_modelo, caminho_vetorizador)

        # Exporta para ONNX
        logger.info(f"\n2. Exportando para ONNX...")
        otimizador.exportar_onnx(caminho_onnx)

        # Verifica se arquivo foi criado
        if os.path.exists(caminho_onnx):
            tamanho_kb = os.path.getsize(caminho_onnx) / 1024
            logger.info(f"   Arquivo ONNX criado com sucesso")
            logger.info(f"   Tamanho: {tamanho_kb:.2f} KB")
            logger.info(f"   Caminho: {caminho_onnx}")

            # Comparação de tamanhos
            tamanho_modelo_kb = os.path.getsize(caminho_modelo) / 1024
            logger.info(f"\n3. Comparação de Tamanhos:")
            logger.info(f"   Modelo original (pickle): {tamanho_modelo_kb:.2f} KB")
            logger.info(f"   Modelo ONNX: {tamanho_kb:.2f} KB")

            logger.info("\n" + "=" * 50)
            logger.info("EXPORTAÇÃO CONCLUÍDA COM SUCESSO!")
            logger.info("=" * 50)
            logger.info(f"\nUse o modelo ONNX em produção para inferência rápida.")
            logger.info(f"Execute 'python benchmark_latency.py' para medir ganhos de performance.")

            return True
        else:
            logger.error(f"\nErro: Arquivo ONNX não foi criado")
            return False

    except Exception as e:
        logger.error(f"\nErro durante exportação ONNX: {e}")
        logger.exception("Traceback completo:")
        return False


if __name__ == "__main__":
    sucesso = main()
    exit(0 if sucesso else 1)
