"""
Script de entrada para treinamento do modelo de classificação de condições médicas.
"""

import logging
from pathlib import Path

import pandas as pd

from src.model.train import ClassificadorTextoMedico

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    Função principal para treinamento do modelo.
    """
    logger.info("=" * 50)
    logger.info("INICIANDO TREINAMENTO DO MODELO")
    logger.info("=" * 50)

    try:
        # Carrega dados
        logger.info("\n1. Carregando dados...")
        df_treino = pd.read_csv("data/raw/medical_tc_train.csv")
        logger.info(f"   Amostras de treino: {df_treino.shape[0]}")
        logger.info(f"   Features: {df_treino.columns.tolist()}")

        # Extrai textos e labels (adapta para colunas do dataset)
        # Colunas do dataset: 'medical_abstract' para texto, 'condition_label' para rótulo
        coluna_texto = "medical_abstract"
        coluna_label = "condition_label"

        textos = df_treino[coluna_texto].tolist()
        labels = df_treino[coluna_label].tolist()

        logger.info(f"   Textos extraídos: {len(textos)}")
        logger.info(f"   Labels únicos: {len(set(labels))}")
        logger.info(f"   Classes: {sorted(set(labels))}")

        # Cria classificador
        logger.info("\n2. Criando classificador...")
        classificador = ClassificadorTextoMedico(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            n_estimadores=150,
            max_depth=25,
            random_state=42,
        )
        logger.info("   Classificador criado com sucesso")

        # Treina modelo
        logger.info("\n3. Treinando modelo...")
        resultado = classificador.treinar(
            textos=textos, labels=labels, tamanho_teste=0.2, tamanho_validacao=0.1
        )

        logger.info("\n4. Resultados do Treinamento:")
        logger.info(f"   Acurácia: {resultado['metricas']['acuracia']:.4f}")
        logger.info(f"   Precisão: {resultado['metricas']['precisao']:.4f}")
        logger.info(f"   Revocação: {resultado['metricas']['revocacao']:.4f}")
        logger.info(f"   F1-Score: {resultado['metricas']['f1']:.4f}")

        # Salva modelo
        logger.info("\n5. Salvando modelo...")
        Path("data/models").mkdir(parents=True, exist_ok=True)
        classificador.salvar("data/models/modelo_medico.pkl", "data/models/vetorizador_medico.pkl")
        logger.info("   Modelo salvo com sucesso")

        logger.info("\n" + "=" * 50)
        logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 50)

        return True

    except FileNotFoundError as e:
        logger.error(f"\nErro: Arquivo não encontrado: {e}")
        logger.error("Certifique-se de que os dados estão em 'data/raw/'")
        return False

    except Exception as e:
        logger.error(f"\nErro durante treinamento: {e}")
        logger.exception("Traceback completo:")
        return False


if __name__ == "__main__":
    sucesso = main()
    exit(0 if sucesso else 1)
