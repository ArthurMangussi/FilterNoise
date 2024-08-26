# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

class PreprocessingDatasets:
    # ------------------------------------------------------------------------
    @staticmethod
    def label_encoder(df:pd.DataFrame, lista_nome_colunas:list):
        data = df.copy()
        le = LabelEncoder()

        for att in lista_nome_colunas:
            data[att] = le.fit_transform(data[att])

        return data
    # ------------------------------------------------------------------------
    @staticmethod
    def ordinal_encoder(df:pd.DataFrame, lista_nome_colunas:list):
        data = df.copy()
        enc = OrdinalEncoder()

        for att in lista_nome_colunas:
            X = np.array(data[att]).reshape(-1, 1)
            data[att] = enc.fit_transform(X)

        return data
    # ------------------------------------------------------------------------
    @staticmethod
    def one_hot_encode(df:pd.DataFrame, lista_nome_colunas:list):
        data = df.copy()
        data_categorical = data[lista_nome_colunas]

        encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = encoder.fit_transform(data_categorical)

        # Criar um novo DataFrame com as colunas one-hot
        columns = encoder.get_feature_names_out(lista_nome_colunas)
        df_encoded = pd.DataFrame(one_hot_encoded, columns=columns)

        # Concatenar o DataFrame original com o DataFrame codificado
        data = pd.concat([data, df_encoded], axis=1)

        # Remover as colunas originais
        data = data.drop(columns=lista_nome_colunas)

        return data
    
    # ------------------------------------------------------------------------
    def inicializa_normalizacao(X_treino: pd.DataFrame) -> MinMaxScaler:
        """
        Função para inicializar MinMaxScaler para normalizar o conjunto de dados com base nos dados de treino

        Args:
            X_treino (pd.DataFrame): O dataset a ser normalizado.

        Returns:
            modelo_norm (MinMaxScaler): O objeto MinMaxScaler ajustado que pode ser usado para normalizar outros conjuntos de dados com base nos dados de treinamento.
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        modelo_norm = scaler.fit(X_treino)

        return modelo_norm

    # ------------------------------------------------------------------------
    def normaliza_dados(modelo_normalizador, X) -> pd.DataFrame:
        """
        Função para normalizar os dados usando um modelo de normalização fornecido.

        Args:
            modelo_normalizador: O modelo de normalização a ser usado para normalizar os dados.
            X: Os dados de entrada a serem normalizados.

        Returns:
            X_norm: Os dados normalizados.

        Example Usage:
        ```python

        # Cria um modelo de normalização
        scaler = MinMaxScaler()

        # Normaliza os dados usando o modelo
        normalized_data = normaliza_dados(scaler, X)
        ```
        """

        X_norm = modelo_normalizador.transform(X)
        X_norm_df = pd.DataFrame(X_norm, columns=X.columns)

        return X_norm_df