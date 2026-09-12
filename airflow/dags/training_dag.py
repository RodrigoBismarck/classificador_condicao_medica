"""
DAG Airflow para pipeline de treinamento de modelo de condições médicas.
"""
import os
from datetime import datetime, timedelta
import logging

if not hasattr(os, "register_at_fork"):
    def _register_at_fork_noop(*args, **kwargs):
        return None

    os.register_at_fork = _register_at_fork_noop

from airflow import DAG
from airflow.decorators import task

try:
    from airflow.utils.dates import days_ago
except Exception:
    def days_ago(dias: int):
        return datetime.now() - timedelta(days=dias)

# Configurar logging
logger = logging.getLogger(__name__)


# Argumentos padrão da DAG
argumentos_padrao = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email': ['contato@exemplo.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Funções de tarefa
@task(task_id="carregar_dados_task")
def carregar_dados():
    """
    Carrega dados de treinamento.
    """
    logger.info("Iniciando carregamento de dados...")
    try:
        import pandas as pd

        df_treino = pd.read_csv('data/raw/medical_tc_train.csv')
        df_teste = pd.read_csv('data/raw/medical_tc_test.csv')
        
        logger.info(f"Dados de treino carregados: {df_treino.shape}")
        logger.info(f"Dados de teste carregados: {df_teste.shape}")
        
        # Salva caminho para próxima tarefa
        return {
            'caminho_treino': 'data/raw/medical_tc_train.csv',
            'caminho_teste': 'data/raw/medical_tc_test.csv',
            'amostras_treino': df_treino.shape[0],
            'amostras_teste': df_teste.shape[0]
        }
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise


@task(task_id="preprocessar_dados_task")
def preprocessar_dados():
    """
    Preprocessa dados para treinamento.
    """
    logger.info("Iniciando pré-processamento de dados...")
    try:
        import pandas as pd
        from src.utils.preprocessing import PreprocessadorTexto
        
        # Carrega dados
        df_treino = pd.read_csv('data/raw/medical_tc_train.csv')
        df_teste = pd.read_csv('data/raw/medical_tc_test.csv')
        
        preprocessador = PreprocessadorTexto()
        
        # Preprocessa textos (coluna de texto: 'medical_abstract')
        logger.info("Preprocessando textos de treino...")
        df_treino['texto_processado'] = df_treino['medical_abstract'].apply(
            lambda x: preprocessador.preprocessar(str(x))
        )
        
        logger.info("Preprocessando textos de teste...")
        df_teste['texto_processado'] = df_teste['medical_abstract'].apply(
            lambda x: preprocessador.preprocessar(str(x))
        )
        
        # Salva dados processados
        df_treino.to_csv('data/processed/medical_tc_train_processed.csv', index=False)
        df_teste.to_csv('data/processed/medical_tc_test_processed.csv', index=False)
        
        logger.info("Dados pré-processados e salvos com sucesso")
        
        return {
            'amostras_treino': df_treino.shape[0],
            'amostras_teste': df_teste.shape[0]
        }
    except Exception as e:
        logger.error(f"Erro ao preprocessar dados: {e}")
        raise


@task(task_id="treinar_modelo_task")
def treinar_modelo():
    """
    Treina o modelo de classificação.
    """
    logger.info("Iniciando treinamento do modelo...")
    try:
        import pandas as pd
        from src.model.train import ClassificadorTextoMedico
        
        # Carrega dados processados
        df_treino = pd.read_csv('data/processed/medical_tc_train_processed.csv')
        
        # Extrai textos e labels (coluna de label: 'condition_label')
        textos = df_treino['texto_processado'].tolist()
        labels = df_treino['condition_label'].tolist()
        
        # Cria e treina modelo
        modelo = ClassificadorTextoMedico(
            max_features=5000,
            ngram_range=(1, 2),
            n_estimadores=150,
            max_depth=25
        )
        
        logger.info("Treinando classificador...")
        resultado_treinamento = modelo.treinar(textos, labels)
        
        logger.info(f"Métricas de treinamento: {resultado_treinamento['metricas']}")
        
        # Salva modelo
        modelo.salvar(
            'data/models/modelo_medico.pkl',
            'data/models/vetorizador_medico.pkl'
        )
        
        logger.info("Modelo treinado e salvo com sucesso")
        
        return resultado_treinamento['metricas']
    except Exception as e:
        logger.error(f"Erro ao treinar modelo: {e}")
        raise


@task(task_id="validar_modelo_task")
def validar_modelo():
    """
    Valida desempenho do modelo em conjunto de teste.
    """
    logger.info("Iniciando validação do modelo...")
    try:
        import pandas as pd
        from src.model.predict import MotorPredição
        from sklearn.metrics import accuracy_score
        
        # Carrega dados de teste
        df_teste = pd.read_csv('data/processed/medical_tc_test_processed.csv')
        
        # Carrega modelo
        motor = MotorPredição(
            'data/models/modelo_medico.pkl',
            'data/models/vetorizador_medico.pkl'
        )
        
        # Faz predições
        labels_reais = df_teste['condition_label'].tolist()
        predicoes = []
        
        for i, texto in enumerate(df_teste['texto_processado']):
            resultado = motor.prever(texto)
            predicoes.append(resultado['classe_predita'])
        
        # Calcula acurácia
        acuracia_teste = accuracy_score(labels_reais, predicoes)
        
        logger.info(f"Acurácia em teste: {acuracia_teste:.4f}")
        
        # Valida se acurácia está acima de limite mínimo
        acuracia_minima = 0.85  # 85%
        if acuracia_teste < acuracia_minima:
            logger.warning(f"Acurácia ({acuracia_teste:.4f}) abaixo do mínimo ({acuracia_minima})")
        
        logger.info("Validação concluída com sucesso")
        return acuracia_teste
    except Exception as e:
        logger.error(f"Erro ao validar modelo: {e}")
        raise


@task(task_id="registrar_conclusao_task")
def registrar_conclusao(metricas, acuracia_teste):
    """
    Registra conclusão do pipeline.
    """
    logger.info("Pipeline de treinamento concluído!")
    
    try:
        logger.info(f"Resumo do pipeline:")
        logger.info(f"  - Metricas de validação: {metricas}")
        logger.info(f"  - Acurácia em teste: {acuracia_teste:.4f}")
        logger.info(f"  - Timestamp: {datetime.now().isoformat()}")
    except Exception as e:
        logger.warning(f"Erro ao recuperar métricas: {e}")


with DAG(
    'pipeline_treinamento_condicoes_medicas',
    default_args=argumentos_padrao,
    description='Pipeline MLOps para treinamento de modelo de classificação de condições médicas',
    schedule='0 2 * * 0',  # Toda segunda-feira às 2:00 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'medical', 'machine-learning'],
) as com_dag:

    tarefa_carregar = carregar_dados()
    tarefa_preprocessar = preprocessar_dados()
    tarefa_treinar = treinar_modelo()
    tarefa_validar = validar_modelo()
    tarefa_registrar = registrar_conclusao(tarefa_treinar, tarefa_validar)

    tarefa_carregar >> tarefa_preprocessar >> tarefa_treinar >> tarefa_validar >> tarefa_registrar
