"""
Testes para módulos de modelo (treinamento, predição, pré-processamento).
"""

import sys
from pathlib import Path

import pytest

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.preprocessing import PreprocessadorTexto  # noqa: E402


class TestPreprocessadorTexto:
    """Testes para a classe PreprocessadorTexto."""

    @pytest.fixture
    def preprocessador(self):
        """Cria instância de PreprocessadorTexto."""
        return PreprocessadorTexto()

    def teste_limpar_texto(self, preprocessador):
        """Testa função de limpeza de texto."""
        texto = "Paciente com FEBRE... e tosse!!!"
        texto_limpo = preprocessador.limpar_texto(texto)

        # Verifica se minúsculas e pontuação foram removidas
        assert texto_limpo.islower()
        assert "..." not in texto_limpo
        assert "!!!" not in texto_limpo

    def teste_remover_stopwords(self, preprocessador):
        """Testa remoção de stopwords."""
        texto = "O paciente apresenta febre alta"
        texto_sem_stopwords = preprocessador.remover_stopwords(texto)

        # Stopwords como "o", "e", "com" devem ser removidos
        palavras = texto_sem_stopwords.split()
        assert "o" not in palavras

    def teste_lematizar(self, preprocessador):
        """Testa lematização de palavras."""
        texto = "apresentando febre febrando"
        texto_lematizado = preprocessador.lematizar(texto)

        # Deve converter para forma lemma
        assert isinstance(texto_lematizado, str)

    def teste_preprocessar_completo(self, preprocessador):
        """Testa pipeline completo de pré-processamento."""
        texto = "O PACIENTE apresenta FEBRE alta e tosse seca!!!"
        texto_processado = preprocessador.preprocessar(texto)

        # Verificações básicas
        assert isinstance(texto_processado, str)
        assert texto_processado.islower()  # Deve estar em minúsculas
        assert "!!!" not in texto_processado  # Pontuação removida

    def teste_preprocessar_lote(self, preprocessador):
        """Testa pré-processamento em lote."""
        textos = ["Paciente com FEBRE", "TOSSE seca prolongada", "Dispneia PROGRESSIVA"]

        textos_processados = preprocessador.preprocessar_lote(textos)

        assert len(textos_processados) == 3
        assert all(isinstance(t, str) for t in textos_processados)

    def teste_texto_vazio(self, preprocessador):
        """Testa processamento de texto vazio."""
        texto = ""
        texto_processado = preprocessador.preprocessar(texto)

        # Deve retornar string vazia ou com espaços mínimos
        assert isinstance(texto_processado, str)

    def teste_caracteres_especiais(self, preprocessador):
        """Testa tratamento de caracteres especiais."""
        texto = "Paciente: @#$%^&*() com sintomas!"
        texto_processado = preprocessador.preprocessar(texto)

        # Caracteres especiais devem ser removidos
        assert "@" not in texto_processado
        assert "#" not in texto_processado

    def teste_acentuacao(self, preprocessador):
        """Testa remoção de acentuação."""
        texto = "Paciente com dor no peito"
        texto_processado = preprocessador.preprocessar(texto)

        # Deve manter estrutura básica
        assert isinstance(texto_processado, str)


class TestIntegracaoPreprocessamento:
    """Testes de integração do pré-processamento."""

    def teste_consitencia_preprocessamento(self):
        """Testa se pré-processamento é consistente."""
        preprocessador = PreprocessadorTexto()

        texto = "Paciente apresenta FEBRE e TOSSE"
        resultado1 = preprocessador.preprocessar(texto)
        resultado2 = preprocessador.preprocessar(texto)

        # Mesmo texto deve sempre dar mesmo resultado
        assert resultado1 == resultado2

    def teste_preservacao_informacoes_importantes(self):
        """Testa se informações médicas importantes são preservadas."""
        preprocessador = PreprocessadorTexto()

        texto = "Paciente com hipertensão arterial sistêmica"
        texto_processado = preprocessador.preprocessar(texto)

        # Palavras-chave médicas devem ser preservadas
        assert "hipertens" in texto_processado or "arterial" in texto_processado

    def teste_lote_grande(self):
        """Testa processamento de lote grande."""
        preprocessador = PreprocessadorTexto()

        # Simula 1000 textos
        textos = [f"Paciente {i} com sintoma {i}" for i in range(1000)]

        # Não deve lançar exceção
        resultado = preprocessador.preprocessar_lote(textos)
        assert len(resultado) == 1000


class TestEdgeCases:
    """Testes para casos extremos (edge cases)."""

    def teste_texto_apenas_numeros(self):
        """Testa processamento de texto com apenas números."""
        preprocessador = PreprocessadorTexto()

        texto = "123 456 789"
        resultado = preprocessador.preprocessar(texto)

        # Deve processar sem erro
        assert isinstance(resultado, str)

    def teste_texto_muito_longo(self):
        """Testa processamento de texto muito longo."""
        preprocessador = PreprocessadorTexto()

        # Texto com 10.000 caracteres
        texto = "Paciente " * 1000
        resultado = preprocessador.preprocessar(texto)

        # Deve processar sem erro
        assert isinstance(resultado, str)
        assert len(resultado) > 0

    def teste_idioma_misto(self):
        """Testa processamento de texto em idiomas mistos."""
        preprocessador = PreprocessadorTexto()

        texto = "Paciente with fever y tosse"
        resultado = preprocessador.preprocessar(texto)

        # Deve processar sem erro
        assert isinstance(resultado, str)


class TestValidacaoSaidaPreprocessamento:
    """Testes para validação da saída do pré-processamento."""

    def teste_saida_uma_string(self):
        """Testa se a saída é sempre uma string."""
        preprocessador = PreprocessadorTexto()
        textos_teste = ["Febre alta", "Tosse seca", "Dyspneia", "A" * 1000]

        for texto in textos_teste:
            resultado = preprocessador.preprocessar(texto)
            assert isinstance(resultado, str), f"Falha para: {texto}"

    def teste_sem_valores_nulos(self):
        """Testa se a saída nunca é None."""
        preprocessador = PreprocessadorTexto()

        textos = ["", "a", "Normal", "!!!"]

        for texto in textos:
            resultado = preprocessador.preprocessar(texto)
            assert resultado is not None, f"Resultado None para: {texto}"

    def teste_sem_whitespace_extra(self):
        """Testa se não há espaços múltiplos consecutivos."""
        preprocessador = PreprocessadorTexto()

        texto = "Paciente  com   FEBRE"
        resultado = preprocessador.preprocessar(texto)

        # Não deve ter múltiplos espaços
        assert "  " not in resultado
