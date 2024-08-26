# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'


import os
import pandas as pd
from scipy.io import arff
from io import StringIO


# Bibliotecas
import pandas as pd
from scipy.stats import norm
import warnings

# Ignorar todos os avisos
warnings.filterwarnings("ignore")


# ==========================================================================
class MyPipeline: 
    # ------------------------------------------------------------------------
    @staticmethod
    def pre_imputed_dataset(data):
        fill_na = {}
        for col_missing in data[data.isna()]:
            media = data[col_missing].mean()
            std = data[col_missing].std()
            tam_sample = data[col_missing].isna().sum()
            index_nan = data[col_missing][data[col_missing].isna()].index

            valores_preencher_miss = norm.rvs(loc=media,
                                            scale=std,
                                            size=tam_sample)
            
            
            dict_nan = dict(zip(index_nan, valores_preencher_miss))
            fill_na[col_missing] = dict_nan
            
        dataset_pre_imputed = data.fillna(fill_na)
        return dataset_pre_imputed     
    # ------------------------------------------------------------------------
    @staticmethod
    def create_dirs(mechanism:str, approach:str):
        """
        Função para criar os diretórios de armazenamento para analisar os resultados.

        Args:
            mechanism (str): nome do mecanismo que os missing serão gerados
            approach (str): Abordagem Multivariado ou Univariado
        """
        os.makedirs(f'./Análises Resultados/Tempos/{mechanism}_{approach}', exist_ok=True)
        os.makedirs(f'./Análises Resultados/Classificação/{mechanism}_{approach}', exist_ok=True)
        os.makedirs(f'./Resultados Parciais Multivariado/{mechanism}_{approach}', exist_ok=True)
        os.makedirs(f"./Análises Resultados/Complexidade/{mechanism}_{approach}/baseline", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
    # ------------------------------------------------------------------------
    @staticmethod
    def cria_dataframe(df:pd.DataFrame) -> pd.DataFrame:
        """
        Função para criar um pandas DataFrame a partir de datasets da biblioteca do sklearn.datasets

        Args:
            df: Um objeto pandas DataFrame.

        Returns:
            Um objeto pandas DataFrame contendo os dados do DataFrame de entrada (df) com uma coluna adicional chamada 'target'.
        """
        dataset = pd.DataFrame(data=df.data, columns=df.feature_names)
        dataset["target"] = df.target
        return dataset

    # ------------------------------------------------------------------------
    @staticmethod
    def split_dataset(dataset: pd.DataFrame, perc_treino: float, perc_teste: float):
        """
        Divide o dataset dado nos conjunto de treino, teste e validação.

        Args:
            dataset (pd.DataFrame): A pandas DataFrame contendo o dataset a ser dividido.
            perc_treino (float): A porcentagem do dataset que será usada para treinamento
            perc_teste (float): A porcentagem do dataset que será usada para teste.

        Returns:
            tuple: A tuple contendo três numpy arrays: (X_treino, X_teste, X_valida)
        """

        dataset = dataset.copy()
        df_shuffle = dataset.sample(frac=1.0, replace=True)

        tamanho_treino = int(perc_treino * len(dataset))
        tamanho_teste = int(perc_teste * len(dataset))

        x_treino = df_shuffle.iloc[:tamanho_treino]
        x_teste = df_shuffle.iloc[tamanho_treino : tamanho_treino + tamanho_teste]
        x_valida = df_shuffle.iloc[tamanho_treino + tamanho_teste :]

        return x_treino, x_teste, x_valida

    # ------------------------------------------------------------------------
    @staticmethod
    def carrega_datasets(path_datasets: str) -> dict:
        """
        Carregue conjuntos de dados de um determinado caminho de diretório e retorne-os como um dicionário.

        Argumentos:
            path_datasets (str): O caminho para o diretório que contém os conjuntos de dados.

        Retorna:
            dict: Um dicionário contendo os conjuntos de dados carregados, onde as chaves são os nomes dos arquivos e os valores são DataFrames do pandas.

        Examplo:
            datasets = carrega_datasets('/path/to/datasets')
            print(datasets)
            # Output: {'dataset1': DataFrame1, 'dataset2': DataFrame2, ...}
        """
        datasets_carregados = {}

        for diretorio, subdiretorios, arquivos in os.walk(path_datasets):
            for nome_arquivo in arquivos:
                caminho_completo = os.path.join(diretorio, nome_arquivo)
                nome, extensao = os.path.splitext(nome_arquivo)

                if nome == "._.DS_Store" or extensao == ".names":
                    continue

                if extensao == ".csv" or extensao == ".data":
                    dados = pd.read_csv(caminho_completo)
                    datasets_carregados[nome] = dados

                elif extensao == ".arff":
                    with open(caminho_completo, "r") as f:
                        data = f.read()

                    buffer_texto = StringIO(data)
                    dados, meta = arff.loadarff(buffer_texto)
                    # Convert the numpy array into a dictionary
                    dados_dict = {name: dados[name] for name in dados.dtype.names}

                    # Decode the values to remove the 'b' prefix from the values
                    dados_decodificados = {
                        k: [x.decode() if isinstance(x, bytes) else x for x in v]
                        for k, v in dados_dict.items()
                    }

                    # Convert the decoded data to a pandas DataFrame
                    df = pd.DataFrame(dados_decodificados)
                    datasets_carregados[nome] = df

                elif extensao == ".xlsx":
                    dados = pd.read_excel(caminho_completo)

                    datasets_carregados[nome] = dados

                else:
                    raise ValueError(f"Formato de arquivo não encontrado: {extensao}")

        return datasets_carregados
    
    # ------------------------------------------------------------------------
    @staticmethod
    def encode_features_categoricas(list_binary_features:list, imputed_dataset:pd.DataFrame)->pd.DataFrame:
        for binary_feature in list_binary_features:
            if binary_feature == "target":
                continue
            imputed_dataset[binary_feature] = [1.0 if valor >= 0.5 else 0.0 for valor in imputed_dataset[binary_feature]]
        return imputed_dataset
    # ------------------------------------------------------------------------
    @staticmethod
    def get_binary_features(data:pd.DataFrame)-> list:
        binary_features = []
        for col in data.columns:
            if len(data[col].unique()) == 2:
                binary_features.append(col)

        return binary_features
    # ------------------------------------------------------------------------
    @staticmethod
    def retorna_dict_types_datasetsFairness() -> dict:
        return {"adult": [True,  True,  False,  True,  True,  True,  False,  False,  False,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True],
                       "ricci": [True, False, False, True, False, True],
                       "german_credit": [True,  False,  False,  True,  True,  False,  True,  False,  True,
                                        False,  False,  True,  True,  True,  True,  True,  True,  True,
                                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True],
                       "bank": [True,  True,  True,  True,  False,  True,  True,  False,  True,
                                False,  False,  False,  False,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True],
                       "credit_card": [False, True, True, True, False, True, True, True, True,
                                        True, True, False, False, False, False, False, False, False,
                                        False, False, False, False, False, True],
                       "student_math": [True,  True,  True,  True,  True,  True,  False,  False,  False,
                                        False,  False,  True,  True,  True,  True,  True,  True,  True,
                                        True,  False,  False,  False,  False,  False,  False,  False,  False,
                                        False,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True],
                       "student_port": [True,  True,  True,  True,  True,  True,  False,  False,  False,
                                        False,  False,  True,  True,  True,  True,  True,  True,  True,
                                        True,  False,  False,  False,  False,  False,  False,  False,  False,
                                        False,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True,  True,  True,  True,  True,  True,  True,  True,  True,
                                        True],
                       "compass_7k": [True,  False,  True,  False,  False,  False,  False,  False,  False,
                                    False,  True,  True,  False,  False,  False,  True,  True,  True,
                                    True,  True,  True,  True,  True],
                       "compass_4k": [True,  False,  True,  False,  False,  False,  False,  False,  False,
                                    False,  True,  True,  False,  False,  False,  True,  True,  True,
                                    True,  True,  True,  True,  True],
                       "law":[False, False, False, False, False, False, True, True, True,
                              True, True, True, True, True, True, True],
                       "diabetes": [True,  False,  True,  True,  True,  False,  False,  False,  False,
                                    False,  False,  False,  True,  True,  True,  False,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True,  True,  True,  True,  True,  True,  True,  True,  True,
                                    True],
                       "kdd": [ False,  True,  True,  True,  False,  True,  True,  True,  True,
                                False,  False,  False,  True,  True,  False,  True,  True,  True,
                                False,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True], 
                       "dutch": [True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True,  True,  True,  True,  True,  True,
                                True,  True,  True,  True]}
    
    # ------------------------------------------------------------------------
    @staticmethod
    def retorna_featuresFairness(nome:str) -> dict:

        if nome == "adult":
            features_protect = ["age","race","sex"]
        elif nome == "ricci":
            features_protect = ["Race"]
        elif nome == "german_credit":
            features_protect = ["age", "personal-status-and-sex"]
        elif nome == "bank":
            features_protect = ["age", "marital"]
        elif nome == "credit_card":
            features_protect = ["sex", "education","marriage"]
        elif nome == "student_math" or nome == "student_port":
            features_protect= ["sex", "age"]
        elif nome == "compass_7k" or nome == "compass_4k":
            features_protect = ["race", "sex"]
        elif nome == "law":
            features_protect = ["race","gender"]
        elif nome == "diabetes":
            features_protect = ["gender"]
        elif nome == "kdd":
            features_protect = ["sex","race"]
        elif nome == "dutch":
            features_protect = ["sex"]

        return features_protect

    

    

    