# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'

import pandas as pd
import numpy as np

from utilsMsc.MyPreprocessing import PreprocessingDatasets

from sklearn.metrics import mean_absolute_error
import warnings

from utilsMsc.MeLogSingle import MeLogger

# Ignorar todos os avisos
warnings.filterwarnings("ignore")


class AnalysisResults:
    def __init__(self) :
        self._logger = MeLogger()    
    # ------------------------------------------------------------------------
    @staticmethod
    def gera_resultado_univa(
        resposta, dataset_normalizado_md, dataset_normalizado_original, missing_id,flag=True
    ):
        
        linhas_nan = dataset_normalizado_md.iloc[:, missing_id][
            dataset_normalizado_md.iloc[:, missing_id].isna()
        ].index
        original = dataset_normalizado_original.copy()

        if flag:
            return mean_absolute_error(
                y_true=original.iloc[linhas_nan, missing_id],
                y_pred=resposta[linhas_nan, missing_id]
            )
        
        else:
            return AnalysisResults.correct_predictions(y_true=original.iloc[linhas_nan, missing_id],
                                                       y_pred=resposta.iloc[linhas_nan, missing_id])

    # ------------------------------------------------------------------------
    @staticmethod
    def gera_resultado_multiva(
        resposta,
        dataset_normalizado_md,
        dataset_normalizado_original,
    ):
        maes = []
        features = dataset_normalizado_md.columns[
            dataset_normalizado_md.isna().any()
        ].tolist()

        for feature in features:
            missing_id = dataset_normalizado_md.columns.get_loc(feature)
            linhas_nan = dataset_normalizado_md.iloc[:, missing_id][
                dataset_normalizado_md.iloc[:, missing_id].isna()
            ].index
            original = dataset_normalizado_original.copy()

            mae = mean_absolute_error(
                y_true=original.iloc[linhas_nan, missing_id],
                y_pred=resposta.iloc[linhas_nan, missing_id],
            )
            maes.append(mae)

        return np.mean(maes), np.std(maes)

    # ------------------------------------------------------------------------
    @staticmethod
    def extrai_resultados(tabela_resultados: dict) -> pd.DataFrame:
        fim = []
        for tags in tabela_resultados.keys():
            if tags not in ["datasets", "nome_datasets", "missing_rate"]:
                model_name, nome_dataset, missing_rate, _, error = tags.split("/")

                erro_teste = tabela_resultados[tags]["teste"]

                fim.append((model_name, nome_dataset, missing_rate, error, erro_teste))

        resultados_final = pd.DataFrame(
            fim,
            columns=["Model", "Dataset", "Missing Rate (%)", "Métrica", "Teste"],
        )
        return resultados_final

    # ------------------------------------------------------------------------
    def calcula_metricas_estatisticas_resultados(
        dataset_resultados: pd.DataFrame, nro_metricas: int, nro_iter: int
    ):
        r = []

        for passo in range(0, len(dataset_resultados), nro_metricas * nro_iter):
            filtrada = dataset_resultados[passo : passo + nro_metricas * nro_iter]

            nome_modelo = filtrada["Model"].unique()[0]
            nome = filtrada["Dataset"].unique()[0]
            md = filtrada["Missing Rate (%)"].unique()[0]
            MAE_mean = filtrada["Teste"][filtrada["Métrica"] == "MAE"].mean()
            MAE_std = filtrada["Teste"][filtrada["Métrica"] == "MAE"].std()

            # self._logger.info(
            #     f"Dataset: {nome} com MD = {md}% MAE = {MAE_mean:.3f} +- {MAE_std:.3f}"
            # )

            r.append(
                (
                    nome_modelo,
                    nome,
                    md,
                    round(MAE_mean, 3),
                    round(MAE_std, 3),
                )
            )

        return pd.DataFrame(
            r,
            columns=[
                "Model",
                "Dataset",
                "Missing Rate",
                "MAE_mean",
                "MAE_std",
            ],
        )

    # ------------------------------------------------------------------------
    def gera_tabela_unificada(self,tipo: str, mecanismo_folder:str, *args):
        """
        Generate a unified table by merging data from multiple CSV files.

        Args:
            tipo (str): Type identifier.
            *args: Variable number of positional arguments (file paths).
            **kwargs: Variable number of keyword arguments.

        Returns:
            pd.DataFrame: The final unified table containing the dataset names, missing rates, and mean and standard deviation of MAE for each model.
        """
        dataframes = []
        model_names = []

        for path in args:
            df = pd.read_csv(
                f".\Resultados Parciais\\{mecanismo_folder}\\{path}.csv", sep=",", index_col=0
            )
            model_name = path.split("_")[0].upper()
            mecanismo = path.split("_")[1]
            dataframes.append(df)
            model_names.append(model_name)

        tabela = {}
        tabela["Dataset"] = dataframes[0]["Dataset"].tolist()
        tabela["Missing Rate"] = dataframes[0]["Missing Rate"].tolist()

        for df, model_name in zip(dataframes, model_names):
            tabela[model_name] = [
                f"{mean} ± {std}" for mean, std in zip(df["MAE_mean"], df["MAE_std"])
            ]

        final = pd.DataFrame(tabela)

        self._logger.info(f"Resultados salvos com sucesso!")
        final.to_csv(f"Resultados Imputacao/{mecanismo_folder}.csv", sep=",", index=False)
        return final

    # ------------------------------------------------------------------------
    @staticmethod
    def heatmap_resultados(path: str, mecanismo: str, estrategia: str):
        all_heatmaps = {}

        resultados = pd.read_csv(path, index_col=0)
        for nome in resultados.Dataset.unique():
            dataset = resultados[resultados.Dataset == nome].reset_index(drop=True)

            dataset = dataset.copy()

            for i in dataset.columns[2:]:
                dataset[i] = dataset[i].str.extract(r"([0-9.]+) ±")

            colunas = {0: "MR = 10%", 1: "MR = 20%", 2: "MR = 40%", 3: "MR = 60%"}

            df = dataset.iloc[:, 2:].T.rename(columns=colunas)
            styled_df = df.style.background_gradient(
                cmap="Grays", subset=["MR = 10%", "MR = 20%", "MR = 40%", "MR = 60%"]
            )
            all_heatmaps[nome] = styled_df

        output = f"Análises Resultados/{mecanismo}-{estrategia}.xlsx"
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for key in all_heatmaps.keys():
                all_heatmaps[key].to_excel(writer, sheet_name=key)

    # ------------------------------------------------------------------------
    def correct_predictions(y_true, y_pred, percentage: bool = False):
        if len(y_true) != len(y_pred):
            raise ValueError("Arrays must have same length")

        correct = sum(a == b for a, b in zip(y_true, y_pred))
        pcp = correct / len(y_true)

        return round(100 * pcp, 2) if percentage else round(pcp, 3)
