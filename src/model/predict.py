"""
Módulo de predição e inferência do classificador de texto médico.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List

import joblib

from src.utils.preprocessing import PreprocessadorTexto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MotorPredição:
    """
    Motor de predição para o classificador de texto médico.
    Especialmente otimizado para servição em produção via API.
    """

    def __init__(self, caminho_modelo: str, caminho_vetorizador: str):
        """
        Inicializa o motor de predição.

        Args:
            caminho_modelo: Caminho do modelo treinado
            caminho_vetorizador: Caminho do vetorizador treinado
        """
        self.caminho_modelo = caminho_modelo
        self.caminho_vetorizador = caminho_vetorizador
        self.preprocessador = PreprocessadorTexto()

        self._carregador_modelo()

    def _carregador_modelo(self):
        """Carrega modelo e vetorizador do disco."""
        try:
            self.classificador = joblib.load(self.caminho_modelo)
            self.vetorizador = joblib.load(self.caminho_vetorizador)
            self.mapa_labels_inverso = joblib.load(
                self.caminho_modelo.replace(".pkl", "_mapa_labels_inverso.pkl")
            )
            self.nomes_condicoes = self._carregar_nomes_condicoes()
            self.info_modelo = self._obter_info_modelo()
            logger.info("Modelo carregado com sucesso")
        except FileNotFoundError as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

    def _carregar_nomes_condicoes(self) -> Dict[int, str]:
        """Carrega o mapeamento de ids para nomes das condições."""
        caminho_labels = Path(self.caminho_modelo).resolve().parent.parent / "raw" / "medical_tc_labels.csv"

        if not caminho_labels.exists():
            logger.warning("Arquivo de labels não encontrado em %s", caminho_labels)
            return {}

        with caminho_labels.open("r", encoding="utf-8", newline="") as arquivo:
            leitor = csv.DictReader(arquivo)
            return {
                int(linha["condition_label"]): linha["condition_name"]
                for linha in leitor
                if linha.get("condition_label") and linha.get("condition_name")
            }

    def _obter_nome_classe(self, id_classe: int) -> str:
        """Resolve o nome textual de uma classe a partir do id previsto."""
        if id_classe in self.nomes_condicoes:
            return self.nomes_condicoes[id_classe]

        return self.mapa_labels_inverso.get(id_classe, str(id_classe))

    def _obter_info_modelo(self) -> Dict:
        """Obtém informações sobre o modelo."""
        try:
            n_features = self.vetorizador.get_feature_names_out().__len__()
        except Exception:
            n_features = 5000  # Valor padrão

        return {
            "n_features": n_features,
            "n_classes": len(self.classificador.classes_),
            "tempo_treino": None,
            "acuracia_teste": None,
        }

    def prever(self, texto: str) -> Dict:
        """
        Prediz classe para um único texto.

        Args:
            texto: Texto médico a classificar

        Returns:
            Dicionário com predição e confiança
        """
        # Pré-processa
        texto_processado = self.preprocessador.preprocessar(texto)

        # Vetoriza
        X = self.vetorizador.transform([texto_processado])

        # Prediz
        id_classe = int(self.classificador.predict(X)[0])
        probabilidades = self.classificador.predict_proba(X)[0]
        classes_modelo = [int(classe) for classe in self.classificador.classes_]
        probabilidades_por_classe = {
            classe: round(float(prob), 4)
            for classe, prob in zip(classes_modelo, probabilidades)
        }

        # Cria resposta
        nome_classe = self._obter_nome_classe(id_classe)
        confianca = probabilidades_por_classe.get(id_classe, round(float(max(probabilidades)), 4))

        return {
            "texto_original": texto,
            "classe_predita": nome_classe,
            "id_classe": id_classe,
            "confianca": round(float(confianca), 4),
            "todas_probabilidades": {
                self._obter_nome_classe(classe): prob
                for classe, prob in probabilidades_por_classe.items()
            },
        }

    def prever_lote(self, textos: List[str]) -> Dict:
        """
        Prediz classes para múltiplos textos.

        Args:
            textos: Lista de textos médicos

        Returns:
            Dicionário com predições em lote
        """
        predicoes = []

        for texto in textos:
            predicao = self.prever(texto)
            predicoes.append(predicao)

        return {"total": len(textos), "predicoes": predicoes}

    def obter_info_modelo(self) -> Dict:
        """
        Retorna informações sobre o modelo.

        Returns:
            Dicionário com informações do modelo
        """
        return {
            "nome": "Classificador de Condições Médicas",
            "tipo": "Random Forest com TF-IDF",
            "versao": "1.0",
            "n_features": self.info_modelo["n_features"],
            "n_classes": self.info_modelo["n_classes"],
            "classes": [self._obter_nome_classe(int(classe)) for classe in self.classificador.classes_],
            "arquitetura": {
                "vetorizador": "TF-IDF (5000 features, 1-2 gramas)",
                "classificador": "Random Forest (150 árvores, max_depth=20)",
            },
        }
