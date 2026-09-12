"""
Utilitários de pré-processamento para classificação de texto médico.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Baixa dados NLTK necessários
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


class PreprocessadorTexto:
    """Lida com pré-processamento de texto para abstratos médicos."""

    def __init__(self):
        self.stopwords_en = set(stopwords.words("english"))
        self.lematizador = WordNetLemmatizer()

    def limpar_texto(self, texto: str) -> str:
        """
        Limpa texto médico removendo caracteres especiais e espaços extras.

        Args:
            texto: String de texto bruto

        Returns:
            Texto limpo
        """
        # Converte para minúsculas
        texto = texto.lower()

        # Remove URLs
        texto = re.sub(r"http\S+|www\S+", "", texto)

        # Remove endereços de email
        texto = re.sub(r"\S+@\S+", "", texto)

        # Remove números e caracteres especiais mas mantém espaços
        texto = re.sub(r"[^a-zA-Z\s]", "", texto)

        # Remove espaços extras
        texto = re.sub(r"\s+", " ", texto).strip()

        return texto

    def remover_stopwords(self, texto: str) -> str:
        """
        Remove stopwords em inglês do texto.

        Args:
            texto: Texto limpo

        Returns:
            Texto sem stopwords
        """
        tokens = word_tokenize(texto)
        tokens_filtrados = [palavra for palavra in tokens if palavra not in self.stopwords_en]
        return " ".join(tokens_filtrados)

    def lematizar(self, texto: str) -> str:
        """
        Lematiza palavras no texto.

        Args:
            texto: Texto a lematizar

        Returns:
            Texto lematizado
        """
        tokens = word_tokenize(texto)
        tokens_lematizados = [self.lematizador.lemmatize(palavra) for palavra in tokens]
        return " ".join(tokens_lematizados)

    def preprocessar(self, texto: str) -> str:
        """
        Aplica o pipeline completo de pré-processamento.

        Args:
            texto: Texto bruto

        Returns:
            Texto totalmente pré-processado
        """
        texto = self.limpar_texto(texto)
        texto = self.remover_stopwords(texto)
        texto = self.lematizar(texto)
        return texto

    def preprocessar_lote(self, textos) -> list:
        """
        Pré-processa um lote de textos usando a instância atual.

        Args:
            textos: Lista de strings de texto

        Returns:
            Lista de textos pré-processados
        """
        return [self.preprocessar(texto) for texto in textos]


def preprocessar_lote(textos, preprocessador=None):
    """
    Pré-processa um lote de textos.

    Args:
        textos: Lista de strings de texto
        preprocessador: Instância de PreprocessadorTexto

    Returns:
        Lista de textos pré-processados
    """
    if preprocessador is None:
        preprocessador = PreprocessadorTexto()

    return [preprocessador.preprocessar(texto) for texto in textos]
