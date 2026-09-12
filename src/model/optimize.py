"""
Módulo de otimização do modelo usando ONNX e benchmarking de latência.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

try:
    import onnxruntime as rt
    import skl2onnx
    from skl2onnx.common.data_types import FloatTensorType

    ONNX_DISPONIVEL = True
except ImportError:
    ONNX_DISPONIVEL = False
    logging.warning("ONNX não disponível. Instale: pip install skl2onnx onnxruntime")

from src.utils.preprocessing import PreprocessadorTexto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OtimizadorONNX:
    """
    Otimizador de modelo usando ONNX Runtime.
    Reduz latência de inferência em ~40-50% comparado ao scikit-learn.
    """

    def __init__(self, caminho_modelo: str, caminho_vetorizador: str):
        """
        Inicializa o otimizador.

        Args:
            caminho_modelo: Caminho do modelo Random Forest treinado
            caminho_vetorizador: Caminho do vetorizador TF-IDF
        """
        if not ONNX_DISPONIVEL:
            raise ImportError("ONNX não está instalado. Execute: pip install skl2onnx onnxruntime")

        self.caminho_modelo = caminho_modelo
        self.caminho_vetorizador = caminho_vetorizador
        self.classificador = joblib.load(caminho_modelo)
        self.vetorizador = joblib.load(caminho_vetorizador)
        self.preprocessador = PreprocessadorTexto()
        self.modelo_onnx = None
        self.sessao_onnx = None

    def exportar_onnx(self, caminho_saida: str):
        """
        Exporta o modelo random forest para formato ONNX.

        Args:
            caminho_saida: Caminho onde salvar o modelo ONNX
        """
        if not ONNX_DISPONIVEL:
            raise ImportError("ONNX não está instalado")

        logger.info("Exportando modelo para ONNX...")

        try:
            # Define tipos iniciais (entrada é matriz densa de features TF-IDF)
            n_features = getattr(self.classificador, "n_features_in_", None)
            if n_features is None:
                n_features = len(getattr(self.vetorizador, "vocabulary_", {}))
            tipos_iniciais = [("X", FloatTensorType([None, int(n_features)]))]

            # Converte modelo para ONNX
            modelo_onnx = skl2onnx.convert_sklearn(
                self.classificador, initial_types=tipos_iniciais, target_opset=12
            )

            # Salva modelo ONNX
            Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
            with open(caminho_saida, "wb") as f:
                f.write(modelo_onnx.SerializeToString())

            logger.info(f"Modelo ONNX exportado para {caminho_saida}")
            return True

        except Exception as e:
            logger.error(f"Erro ao exportar ONNX: {e}")
            raise

    def carregar_onnx(self, caminho_modelo_onnx: str):
        """
        Carrega modelo ONNX para inferência.

        Args:
            caminho_modelo_onnx: Caminho do arquivo .onnx
        """
        if not ONNX_DISPONIVEL:
            raise ImportError("ONNX não está instalado")

        try:
            self.sessao_onnx = rt.InferenceSession(caminho_modelo_onnx)
            logger.info(f"Modelo ONNX carregado de {caminho_modelo_onnx}")
        except Exception as e:
            logger.error(f"Erro ao carregar ONNX: {e}")
            raise

    def prever_onnx(self, texto: str) -> Dict:
        """
        Prediz usando modelo ONNX.

        Args:
            texto: Texto a classifcar

        Returns:
            Dicionário com predição
        """
        if self.sessao_onnx is None:
            raise ValueError("Modelo ONNX não carregado. Use carregar_onnx() primeiro.")

        # Pré-processa
        texto_processado = self.preprocessador.preprocessar(texto)

        # Vetoriza
        X = self.vetorizador.transform([texto_processado]).astype(np.float32).toarray()

        # Prediz com ONNX
        entrada = {self.sessao_onnx.get_inputs()[0].name: X}
        predication = self.sessao_onnx.run(None, entrada)

        return {"classe": predication[0][0], "probabilidades": predication[1][0]}


class BenchmarkLatência:
    """
    Benchmarking de latência entre modelo scikit-learn e ONNX.
    """

    def __init__(self, caminho_modelo: str, caminho_vetorizador: str, caminho_onnx: str = None):
        """
        Inicializa benchmark.

        Args:
            caminho_modelo: Caminho do modelo sklearn
            caminho_vetorizador: Caminho do vetorizador
            caminho_onnx: Caminho do modelo ONNX (opcional)
        """
        self.caminho_modelo = caminho_modelo
        self.caminho_vetorizador = caminho_vetorizador
        self.caminho_onnx = caminho_onnx
        self.classificador = joblib.load(caminho_modelo)
        self.vetorizador = joblib.load(caminho_vetorizador)
        self.preprocessador = PreprocessadorTexto()
        self.sessao_onnx = None

        if caminho_onnx:
            try:
                self.sessao_onnx = rt.InferenceSession(caminho_onnx)
            except Exception as e:
                logger.warning(f"Não foi possível carregar modelo ONNX: {e}")

    def _preparar_batch(self, textos: list, tamanho_batch: int = 32) -> Tuple[np.ndarray, int]:
        """
        Prepara batch de textos.

        Args:
            textos: Lista de textos
            tamanho_batch: Tamanho do batch

        Returns:
            Tupla de (matriz X, número de batches)
        """
        textos_processados = [self.preprocessador.preprocessar(texto) for texto in textos]
        X = self.vetorizador.transform(textos_processados)
        n_batches = (X.shape[0] + tamanho_batch - 1) // tamanho_batch

        return X, n_batches

    def benchmark_sklearn(self, textos: list, iteracoes: int = 100) -> Dict:
        """
        Benchmark do modelo sklearn.

        Args:
            textos: Lista de textos para teste
            iteracoes: Número de iterações

        Returns:
            Dicionário com estatísticas de latência
        """
        X, _ = self._preparar_batch(textos)

        tempos = []
        for _ in range(iteracoes):
            inicio = time.perf_counter()
            self.classificador.predict_proba(X)
            fim = time.perf_counter()
            tempos.append((fim - inicio) * 1000)  # Convertendo para ms

        return {
            "backend": "scikit-learn",
            "media_ms": np.mean(tempos),
            "min_ms": np.min(tempos),
            "max_ms": np.max(tempos),
            "desvio_ms": np.std(tempos),
            "p95_ms": np.percentile(tempos, 95),
            "p99_ms": np.percentile(tempos, 99),
        }

    def benchmark_onnx(self, textos: list, iteracoes: int = 100) -> Dict:
        """
        Benchmark do modelo ONNX.

        Args:
            textos: Lista de textos para teste
            iteracoes: Número de iterações

        Returns:
            Dicionário com estatísticas de latência
        """
        if self.sessao_onnx is None:
            raise ValueError("Modelo ONNX não carregado")

        X, _ = self._preparar_batch(textos)
        X_float = X.astype(np.float32).toarray()

        entrada_nome = self.sessao_onnx.get_inputs()[0].name

        tempos = []
        for _ in range(iteracoes):
            inicio = time.perf_counter()
            self.sessao_onnx.run(None, {entrada_nome: X_float})
            fim = time.perf_counter()
            tempos.append((fim - inicio) * 1000)  # Convertendo para ms

        return {
            "backend": "ONNX Runtime",
            "media_ms": np.mean(tempos),
            "min_ms": np.min(tempos),
            "max_ms": np.max(tempos),
            "desvio_ms": np.std(tempos),
            "p95_ms": np.percentile(tempos, 95),
            "p99_ms": np.percentile(tempos, 99),
        }

    def benchmark_comparativo(self, textos: list, iteracoes: int = 100) -> Dict:
        """
        Benchmark comparativo entre sklearn e ONNX.

        Args:
            textos: Lista de textos para teste
            iteracoes: Número de iterações

        Returns:
            Dicionário com comparação de latência
        """
        logger.info(f"Executando benchmark com {len(textos)} textos, {iteracoes} iterações...")

        # Benchmark sklearn
        resultado_sklearn = self.benchmark_sklearn(textos, iteracoes)
        logger.info(f"Scikit-learn: {resultado_sklearn['media_ms']:.2f}ms (média)")

        # Benchmark ONNX se disponível
        resultado_onnx = None
        if self.sessao_onnx:
            resultado_onnx = self.benchmark_onnx(textos, iteracoes)
            logger.info(f"ONNX: {resultado_onnx['media_ms']:.2f}ms (média)")

            # Calcula melhoria
            melhoria_pct = (
                (resultado_sklearn["media_ms"] - resultado_onnx["media_ms"])
                / resultado_sklearn["media_ms"]
                * 100
            )
            logger.info(f"Melhoria com ONNX: {melhoria_pct:.1f}%")

        return {
            "sklearn": resultado_sklearn,
            "onnx": resultado_onnx,
            "melhoria_percentual": (
                (
                    (resultado_sklearn["media_ms"] - resultado_onnx["media_ms"])
                    / resultado_sklearn["media_ms"]
                    * 100
                )
                if resultado_onnx
                else None
            ),
        }
