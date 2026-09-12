"""
Script de benchmark de latência entre modelo scikit-learn e ONNX.
"""
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from src.model.optimize import BenchmarkLatência
except ImportError:
    print("Erro: ONNX não está instalado. Instale com: pip install skl2onnx onnxruntime")
    exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Executa benchmark de latência e salva resultados.
    """
    logger.info("="*60)
    logger.info("BENCHMARK DE LATÊNCIA: SCIKIT-LEARN vs ONNX")
    logger.info("="*60)
    
    try:
        # Caminhos dos arquivos
        caminho_modelo = 'data/models/modelo_medico.pkl'
        caminho_vetorizador = 'data/models/vetorizador_medico.pkl'
        caminho_onnx = 'data/models/modelo_medico.onnx'
        
        # Verifica se arquivo de modelo existe
        if not os.path.exists(caminho_modelo):
            logger.error(f"\nErro: Modelo não encontrado em {caminho_modelo}")
            logger.error("Execute 'python train_model.py' para treinar um modelo")
            return False
        
        # Carrega dados de teste
        logger.info(f"\n1. Carregando dados de teste...")
        try:
            df_teste = pd.read_csv('data/raw/medical_tc_test.csv')
            coluna_texto = 'medical_abstract'
            textos_teste = df_teste[coluna_texto].head(100).tolist()  # Usa primeiras 100 amostras
            logger.info(f"   Amostras carregadas: {len(textos_teste)}")
        except Exception as e:
            logger.warning(f"   Não foi possível carregar dados reais: {e}")
            logger.warning("   Usando textos de exemplo para benchmark...")
            textos_teste = [
                "Paciente apresenta febre alta e tosse seca" * 5 for _ in range(100)
            ]
        
        # Benchmark sem ONNX (apenas sklearn)
        logger.info(f"\n2. Executando benchmark SCIKIT-LEARN...")
        benchmark = BenchmarkLatência(
            caminho_modelo=caminho_modelo,
            caminho_vetorizador=caminho_vetorizador,
            caminho_onnx=None
        )
        
        resultado_sklearn = benchmark.benchmark_sklearn(
            textos=textos_teste,
            iteracoes=50
        )
        
        # Tenta benchmark com ONNX se arquivo existe
        resultado_onnx = None
        melhoria_pct = None
        
        if os.path.exists(caminho_onnx):
            logger.info(f"\n3. Executando benchmark ONNX...")
            benchmark_onnx = BenchmarkLatência(
                caminho_modelo=caminho_modelo,
                caminho_vetorizador=caminho_vetorizador,
                caminho_onnx=caminho_onnx
            )
            
            resultado_onnx = benchmark_onnx.benchmark_onnx(
                textos=textos_teste,
                iteracoes=50
            )
            
            # Calcula melhoria
            melhoria_pct = (
                (resultado_sklearn['media_ms'] - resultado_onnx['media_ms']) /
                resultado_sklearn['media_ms'] * 100
            )
        else:
            logger.warning(f"\n3. Modelo ONNX não encontrado em {caminho_onnx}")
            logger.warning("   Execute 'python export_onnx_model.py' para gerar modelo ONNX")
        
        # Exibe resultados
        logger.info("\n" + "="*60)
        logger.info("RESULTADOS DO BENCHMARK")
        logger.info("="*60)
        
        logger.info(f"\nSCIKIT-LEARN:")
        logger.info(f"  Média: {resultado_sklearn['media_ms']:.4f} ms")
        logger.info(f"  Mín: {resultado_sklearn['min_ms']:.4f} ms")
        logger.info(f"  Máx: {resultado_sklearn['max_ms']:.4f} ms")
        logger.info(f"  Desvio padrão: {resultado_sklearn['desvio_ms']:.4f} ms")
        logger.info(f"  P95: {resultado_sklearn['p95_ms']:.4f} ms")
        logger.info(f"  P99: {resultado_sklearn['p99_ms']:.4f} ms")
        
        if resultado_onnx:
            logger.info(f"\nONNX RUNTIME:")
            logger.info(f"  Média: {resultado_onnx['media_ms']:.4f} ms")
            logger.info(f"  Mín: {resultado_onnx['min_ms']:.4f} ms")
            logger.info(f"  Máx: {resultado_onnx['max_ms']:.4f} ms")
            logger.info(f"  Desvio padrão: {resultado_onnx['desvio_ms']:.4f} ms")
            logger.info(f"  P95: {resultado_onnx['p95_ms']:.4f} ms")
            logger.info(f"  P99: {resultado_onnx['p99_ms']:.4f} ms")
            
            logger.info(f"\nMELHORIA COM ONNX:")
            logger.info(f"  {melhoria_pct:.2f}% mais rápido (latência)")
            logger.info(f"  Razão de speedup: {resultado_sklearn['media_ms'] / resultado_onnx['media_ms']:.2f}x")
        
        # Salva relatório
        logger.info(f"\n4. Salvando relatório...")
        
        relatorio = {
            'sklearn': resultado_sklearn,
            'onnx': resultado_onnx,
            'melhoria_percentual': melhoria_pct
        }
        
        import json
        with open('data/models/benchmark_relatorio.json', 'w') as f:
            # Converter arrays numpy para listas para JSON serialization
            relatorio_json = {}
            for backend, metricas in [('sklearn', resultado_sklearn), ('onnx', resultado_onnx)]:
                if metricas:
                    relatorio_json[backend] = {k: float(v) if isinstance(v, np.floating) else v 
                                               for k, v in metricas.items()}
            relatorio_json['melhoria_percentual'] = float(melhoria_pct) if melhoria_pct else None
            
            json.dump(relatorio_json, f, indent=2)
        
        logger.info(f"   Relatório salvo em 'data/models/benchmark_relatorio.json'")
        
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK CONCLUÍDO COM SUCESSO!")
        logger.info("="*60)
        
        return True
    
    except FileNotFoundError as e:
        logger.error(f"\nErro: Arquivo não encontrado: {e}")
        return False
    
    except Exception as e:
        logger.error(f"\nErro durante benchmark: {e}")
        logger.exception("Traceback completo:")
        return False


if __name__ == "__main__":
    sucesso = main()
    exit(0 if sucesso else 1)
