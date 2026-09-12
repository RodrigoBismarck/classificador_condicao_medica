"""
DAG Airflow para pipeline de treinamento de modelo de condições médicas.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import logging

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
def carregar_dados():
    """
    Carrega dados de treinamento.
    """
    logger.info("Iniciando carregamento de dados...")
    try:
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


def preprocessar_dados(**contexto):
    """
    Preprocessa dados para treinamento.
    """
    logger.info("Iniciando pré-processamento de dados...")
    try:
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


def treinar_modelo(**contexto):
    """
    Treina o modelo de classificação.
    """
    logger.info("Iniciando treinamento do modelo...")
    try:
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
        
        # Retorna métricas
        contexto['task_instance'].xcom_push(
            key='metricas',
            value=resultado_treinamento['metricas']
        )
        
        return resultado_treinamento['metricas']
    except Exception as e:
        logger.error(f"Erro ao treinar modelo: {e}")
        raise


def validar_modelo(**contexto):
    """
    Valida desempenho do modelo em conjunto de teste.
    """
    logger.info("Iniciando validação do modelo...")
    try:
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
            _, nome_classe, _ = motor.prever(texto)
            predicoes.append(nome_classe)
        
        # Calcula acurácia
        acuracia_teste = accuracy_score(labels_reais, predicoes)
        
        logger.info(f"Acurácia em teste: {acuracia_teste:.4f}")
        
        # Valida se acurácia está acima de limite mínimo
        acuracia_minima = 0.85  # 85%
        if acuracia_teste < acuracia_minima:
            logger.warning(f"Acurácia ({acuracia_teste:.4f}) abaixo do mínimo ({acuracia_minima})")
        
        contexto['task_instance'].xcom_push(
            key='acuracia_teste',
            value=acuracia_teste
        )
        
        logger.info("Validação concluída com sucesso")
        return acuracia_teste
    except Exception as e:
        logger.error(f"Erro ao validar modelo: {e}")
        raise


def registrar_conclusao(**contexto):
    """
    Registra conclusão do pipeline.
    """
    logger.info("Pipeline de treinamento concluído!")
    
    # Recupera métricas dos steps anteriores
    ti = contexto['task_instance']
    
    try:
        metricas = ti.xcom_pull(task_ids='treinar_modelo_task', key='metricas')
        acuracia_teste = ti.xcom_pull(task_ids='validar_modelo_task', key='acuracia_teste')
        
        logger.info(f"Resumo do pipeline:")
        logger.info(f"  - Metricas de validação: {metricas}")
        logger.info(f"  - Acurácia em teste: {acuracia_teste:.4f}")
        logger.info(f"  - Timestamp: {datetime.now().isoformat()}")
    except Exception as e:
        logger.warning(f"Erro ao recuperar métricas: {e}")


# Criar DAG
com_dag = DAG(
    'pipeline_treinamento_condicoes_medicas',
    default_args=argumentos_padrao,
    description='Pipeline MLOps para treinamento de modelo de classificação de condições médicas',
    schedule_interval='0 2 * * 0',  # Toda segunda-feira às 2:00 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'medical', 'machine-learning'],
)


# Definir tarefas
tarefa_carregar = PythonOperator(
    task_id='carregar_dados_task',
    python_callable=carregar_dados,
    dag=com_dag,
)

tarefa_preprocessar = PythonOperator(
    task_id='preprocessar_dados_task',
    python_callable=preprocessar_dados,
    dag=com_dag,
)

tarefa_treinar = PythonOperator(
    task_id='treinar_modelo_task',
    python_callable=treinar_modelo,
    dag=com_dag,
)

tarefa_validar = PythonOperator(
    task_id='validar_modelo_task',
    python_callable=validar_modelo,
    dag=com_dag,
)

tarefa_registrar = PythonOperator(
    task_id='registrar_conclusao_task',
    python_callable=registrar_conclusao,
    dag=com_dag,
)


# Definir dependências (pipeline linear)
tarefa_carregar >> tarefa_preprocessar >> tarefa_treinar >> tarefa_validar >> tarefa_registrar
