"""
Testes para a API FastAPI de classificação de condições médicas.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app


# Fixture para client
@pytest.fixture
def cliente():
    """Cria um cliente de teste FastAPI."""
    return TestClient(app)


class TestEndpointVitalis:
    """Testes para endpoint de vitalidade."""
    
    def teste_verificar_saude(self, cliente):
        """Testa se endpoint /health retorna status ok."""
        resposta = cliente.get("/health")
        assert resposta.status_code == 200
        dados = resposta.json()
        assert "status" in dados
        assert dados["status"] in ["ok", "model_not_loaded"]
        assert "timestamp" in dados
        assert "versao_modelo" in dados


class TestEndpointPredição:
    """Testes para endpoints de predição."""
    
    @patch('src.api.main.motor_predicao')
    def teste_prever_sucesso(self, mock_motor, cliente):
        """Testa predição bem-sucedida."""
        # Mock do motor de predição
        mock_motor.prever.return_value = {
            'texto_original': 'Teste',
            'classe_predita': 'Doença X',
            'id_classe': 0,
            'confianca': 0.95,
            'todas_probabilidades': {'Doença X': 0.95, 'Doença Y': 0.05}
        }
        
        resposta = cliente.post(
            "/predict",
            json={"texto": "Paciente com febre"}
        )
        
        assert resposta.status_code == 200
        dados = resposta.json()
        assert dados['classe_predita'] == 'Doença X'
        assert dados['confianca'] == 0.95
    
    def teste_prever_sem_modelo(self, cliente):
        """Testa erro quando modelo não está carregado."""
        with patch('src.api.main.motor_predicao', None):
            resposta = cliente.post(
                "/predict",
                json={"texto": "Teste"}
            )
            assert resposta.status_code == 503
    
    def teste_prever_texto_vazio(self, cliente):
        """Testa validação de texto vazio."""
        resposta = cliente.post(
            "/predict",
            json={"texto": ""}
        )
        assert resposta.status_code == 422  # Erro de validação
    
    def teste_prever_texto_muito_longo(self, cliente):
        """Testa validação de texto muito longo."""
        texto_longo = "a" * 6000
        resposta = cliente.post(
            "/predict",
            json={"texto": texto_longo}
        )
        assert resposta.status_code == 422  # Erro de validação


class TestEndpointPredçãoLote:
    """Testes para endpoint de predição em lote."""
    
    @patch('src.api.main.motor_predicao')
    def teste_prever_lote_sucesso(self, mock_motor, cliente):
        """Testa predição em lote bem-sucedida."""
        mock_motor.prever_lote.return_value = {
            'total': 2,
            'predicoes': [
                {
                    'texto_original': 'Texto 1',
                    'classe_predita': 'Doença A',
                    'id_classe': 0,
                    'confianca': 0.9,
                    'todas_probabilidades': {'Doença A': 0.9, 'Doença B': 0.1}
                },
                {
                    'texto_original': 'Texto 2',
                    'classe_predita': 'Doença B',
                    'id_classe': 1,
                    'confianca': 0.85,
                    'todas_probabilidades': {'Doença A': 0.15, 'Doença B': 0.85}
                }
            ]
        }
        
        resposta = cliente.post(
            "/predict-batch",
            json={"textos": ["Texto 1", "Texto 2"]}
        )
        
        assert resposta.status_code == 200
        dados = resposta.json()
        assert dados['total'] == 2
        assert len(dados['predicoes']) == 2
    
    def teste_prever_lote_sem_textos(self, cliente):
        """Testa erro quando lista de textos está vazia."""
        resposta = cliente.post(
            "/predict-batch",
            json={"textos": []}
        )
        assert resposta.status_code == 422  # Erro de validação


class TestEndpointInfoModelo:
    """Testes para endpoint de informações do modelo."""
    
    @patch('src.api.main.motor_predicao')
    def teste_obter_info_modelo(self, mock_motor, cliente):
        """Testa obtenção de informações do modelo."""
        mock_motor.obter_info_modelo.return_value = {
            'nome': 'Classificador de Condições Médicas',
            'tipo': 'Random Forest com TF-IDF',
            'versao': '1.0',
            'n_features': 5000,
            'n_classes': 5,
            'classes': ['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4', 'Classe 5'],
            'arquitetura': {
                'vetorizador': 'TF-IDF (5000 features, 1-2 gramas)',
                'classificador': 'Random Forest (150 árvores, max_depth=20)'
            }
        }
        
        resposta = cliente.get("/model-info")
        
        assert resposta.status_code == 200
        dados = resposta.json()
        assert dados['nome'] == 'Classificador de Condições Médicas'
        assert dados['n_classes'] == 5


class TestEndpointMetricas:
    """Testes para endpoint de métricas Prometheus."""
    
    def teste_obter_metricas(self, cliente):
        """Testa se endpoint /metrics retorna dados Prometheus."""
        resposta = cliente.get("/metrics")
        
        assert resposta.status_code == 200
        # Verifica se é formato Prometheus (texto)
        assert isinstance(resposta.text, str)
        assert "#" in resposta.text  # Comentários do Prometheus


class TestValidacaoModelos:
    """Testes de validação dos modelos Pydantic."""
    
    def teste_validacao_requisicao_predicao(self, cliente):
        """Testa validação do modelo RequisicaoPredição."""
        # Teste com campo obrigatório faltando
        resposta = cliente.post(
            "/predict",
            json={}
        )
        assert resposta.status_code == 422
    
    def teste_validacao_requisicao_lote(self, cliente):
        """Testa validação do modelo RequisicaoPredçãoLote."""
        # Teste com campo obrigatório faltando
        resposta = cliente.post(
            "/predict-batch",
            json={}
        )
        assert resposta.status_code == 422


class TestErrosSistema:
    """Testes para tratamento de erros do sistema."""
    
    @patch('src.api.main.motor_predicao')
    def teste_erro_interno_predicao(self, mock_motor, cliente):
        """Testa tratamento de erro interno durante predição."""
        mock_motor.prever.side_effect = Exception("Erro interno")
        
        resposta = cliente.post(
            "/predict",
            json={"texto": "Teste"}
        )
        
        assert resposta.status_code == 500
        dados = resposta.json()
        assert "detail" in dados
