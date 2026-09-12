"""
Módulo de treinamento do classificador de texto médico.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_score,
    recall_score,
    f1_score
)
import logging

from src.utils.preprocessing import PreprocessadorTexto

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificadorTextoMedico:
    """
    Classificador de texto médico usando vetorização TF-IDF e Random Forest.
    
    Este modelo leve é otimizado para:
    - Inferência rápida (adequada para servição de API em tempo real)
    - Pequeno footprint de memória
    - Fácil implantação em containers
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        n_estimadores: int = 100,
        max_depth: int = 20,
        random_state: int = 42
    ):
        """
        Inicializa o classificador.
        
        Args:
            max_features: Número máximo de features para TF-IDF
            ngram_range: Intervalo de n-gramas para TF-IDF
            min_df: Frequência mínima de documentos para TF-IDF
            max_df: Frequência máxima de documentos para TF-IDF
            n_estimadores: Número de árvores em Random Forest
            max_depth: Profundidade máxima das árvores
            random_state: Random state para reprodutibilidade
        """
        self.vetorizador = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.classificador = RandomForestClassifier(
            n_estimators=n_estimadores,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        self.preprocessador = PreprocessadorTexto()
        self.mapa_labels = {}
        self.mapa_labels_inverso = {}
        self.metricas = {}
        
    def _preparar_dados(self, textos: list, labels: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara textos e labels.
        
        Args:
            textos: Lista de strings de texto
            labels: Lista de labels (opcional)
            
        Returns:
            Tupla de (textos vetorizados, labels processados)
        """
        # Pré-processa textos
        textos_processados = [self.preprocessador.preprocessar(texto) for texto in textos]
        
        # Vetoriza
        if labels is None:
            X = self.vetorizador.transform(textos_processados)
        else:
            X = self.vetorizador.fit_transform(textos_processados) if not hasattr(self.vetorizador, 'vocabulary_') else self.vetorizador.transform(textos_processados)
        
        if labels is not None:
            # Mapeia labels string para inteiros se necessário
            if isinstance(labels[0], str):
                y = np.array([self.mapa_labels.get(label, 0) for label in labels])
            else:
                y = np.array(labels)
            return X, y
        
        return X, None
    
    def treinar(
        self,
        textos: list,
        labels: list,
        tamanho_teste: float = 0.2,
        tamanho_validacao: float = 0.1
    ) -> Dict:
        """
        Treina o classificador.
        
        Args:
            textos: Lista de textos de treinamento
            labels: Lista de labels de treinamento
            tamanho_teste: Proporção do conjunto de teste
            tamanho_validacao: Proporção de validação do conjunto de teste
            
        Returns:
            Dicionário com métricas de treinamento
        """
        logger.info("Iniciando treinamento do modelo...")
        
        # Cria mapeamento de labels
        labels_unicos = sorted(set(labels))
        self.mapa_labels = {label: idx for idx, label in enumerate(labels_unicos)}
        self.mapa_labels_inverso = {idx: str(label) for label, idx in self.mapa_labels.items()}
        
        logger.info(f"Mapeamento de labels: {self.mapa_labels}")
        
        # Prepara dados
        X, y = self._preparar_dados(textos, labels)
        
        # Divide dados
        X_treino, X_temp, y_treino, y_temp = train_test_split(
            X, y, test_size=(tamanho_teste + tamanho_validacao), random_state=42
        )
        
        X_val, X_teste, y_val, y_teste = train_test_split(
            X_temp, y_temp, 
            test_size=tamanho_teste / (tamanho_teste + tamanho_validacao),
            random_state=42
        )
        
        logger.info(f"Tamanho conjunto treino: {X_treino.shape[0]}")
        logger.info(f"Tamanho conjunto validação: {X_val.shape[0]}")
        logger.info(f"Tamanho conjunto teste: {X_teste.shape[0]}")
        
        # Treina classificador
        logger.info("Treinando classificador Random Forest...")
        self.classificador.fit(X_treino, y_treino)
        logger.info("Treinamento concluído!")
        
        # Avalia no conjunto de teste
        y_pred = self.classificador.predict(X_teste)
        
        self.metricas = {
            'acuracia': accuracy_score(y_teste, y_pred),
            'precisao': precision_score(y_teste, y_pred, average='weighted', zero_division=0),
            'revocacao': recall_score(y_teste, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_teste, y_pred, average='weighted', zero_division=0)
        }
        
        logger.info("\n=== Relatório de Classificação ===")
        logger.info(classification_report(y_teste, y_pred, 
                                         target_names=[self.mapa_labels_inverso[i] for i in sorted(self.mapa_labels_inverso.keys())]))
        logger.info(f"\nMétricas: {self.metricas}")
        
        return {
            'metricas': self.metricas,
            'relatorio_classificacao': classification_report(y_teste, y_pred, 
                                                          target_names=[self.mapa_labels_inverso[i] for i in sorted(self.mapa_labels_inverso.keys())],
                                                          output_dict=True)
        }
    
    def prever(self, texto: str) -> Tuple[int, str, float]:
        """
        Prediz classe para um único texto.
        
        Args:
            texto: Texto médico a classificar
            
        Returns:
            Tupla de (id_label, nome_label, confianca)
        """
        # Pré-processa
        texto_processado = self.preprocessador.preprocessar(texto)
        
        # Vetoriza
        X = self.vetorizador.transform([texto_processado])
        
        # Prediz
        id_label_pred = self.classificador.predict(X)[0]
        proba_pred = self.classificador.predict_proba(X)[0]
        confianca = float(np.max(proba_pred))
        
        nome_label_pred = self.mapa_labels_inverso[id_label_pred]
        
        return id_label_pred, nome_label_pred, confianca
    
    def prever_lote(self, textos: list) -> list:
        """
        Prediz classes para múltiplos textos.
        
        Args:
            textos: Lista de textos médicos
            
        Returns:
            Lista de predições
        """
        predicoes = []
        for texto in textos:
            id_label, nome_label, confianca = self.prever(texto)
            predicoes.append({
                'id_label': int(id_label),
                'nome_label': nome_label,
                'confianca': confianca
            })
        return predicoes
    
    def salvar(self, caminho_modelo: str, caminho_vetorizador: str):
        """
        Salva modelo e vetorizador em disco.
        
        Args:
            caminho_modelo: Caminho para salvar o modelo
            caminho_vetorizador: Caminho para salvar o vetorizador
        """
        Path(caminho_modelo).parent.mkdir(parents=True, exist_ok=True)
        Path(caminho_vetorizador).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.classificador, caminho_modelo)
        joblib.dump(self.vetorizador, caminho_vetorizador)
        joblib.dump(self.mapa_labels, caminho_modelo.replace('.pkl', '_mapa_labels.pkl'))
        joblib.dump(self.mapa_labels_inverso, caminho_modelo.replace('.pkl', '_mapa_labels_inverso.pkl'))
        
        logger.info(f"Modelo salvo em {caminho_modelo}")
        logger.info(f"Vetorizador salvo em {caminho_vetorizador}")
    
    @classmethod
    def carregar(cls, caminho_modelo: str, caminho_vetorizador: str) -> 'ClassificadorTextoMedico':
        """
        Carrega modelo e vetorizador treinados do disco.
        
        Args:
            caminho_modelo: Caminho do modelo
            caminho_vetorizador: Caminho do vetorizador
            
        Returns:
            Instância de ClassificadorTextoMedico carregada
        """
        instancia = cls()
        instancia.classificador = joblib.load(caminho_modelo)
        instancia.vetorizador = joblib.load(caminho_vetorizador)
        instancia.mapa_labels = joblib.load(caminho_modelo.replace('.pkl', '_mapa_labels.pkl'))
        instancia.mapa_labels_inverso = joblib.load(caminho_modelo.replace('.pkl', '_mapa_labels_inverso.pkl'))
        
        logger.info(f"Modelo carregado de {caminho_modelo}")
        return instancia
