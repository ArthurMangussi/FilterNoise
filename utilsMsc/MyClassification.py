import pandas as pd
import os 

from utilsMsc.MeLogSingle import MeLogger


class ClassificationDatasets:

    def __init__(self):

        self._logger = MeLogger()
    # ------------------------------------------------------------------------
    @staticmethod
    def calcula_dif(df_base,df:pd.DataFrame)->pd.DataFrame:
        data = df.copy()
        for cont in range(len(data.dataset.unique())):
            data.loc[
                data.dataset == df_base.loc[cont, 'dataset'], 'accuracy'
            ] -= df_base.loc[cont, 'accuracy']
            data.loc[
                data.dataset == df_base.loc[cont, 'dataset'], 'recall'
            ] -= df_base.loc[cont, 'recall']
            data.loc[
                data.dataset == df_base.loc[cont, 'dataset'], 'precision'
            ] -= df_base.loc[cont, 'precision']
            data.loc[
                data.dataset == df_base.loc[cont, 'dataset'], 'f1-score'
            ] -= df_base.loc[cont, 'f1-score']

        return data

    # ------------------------------------------------------------------------
    @staticmethod
    def rename_col(df:pd.DataFrame, nome:str)->pd.DataFrame:
        return df.rename(
            columns={
                'accuracy': f'{nome}_accuracy',
                'recall': f'{nome}_recall',
                'precision': f'{nome}_precision',
                'f1-score':f'{nome}_f1-score'
            }
        )
    
    # ------------------------------------------------------------------------
    def calculate_f1(self,mecanismo:str, imputer:str=None, approach:str=None, extension:str=None, baseline:bool=False):
        """
        Calcula a métrica f1-score para os imputes

        Args:
            mecanismo (str): Nome da pasta que contém os arquivos csv de resultados da classificação
            imputer (str): Nome do imputador
            baseline (bool, optional): Indica se o cálculo é para a linha de base. Padrão é False.
        """
        path_arquivos_csv_classificacao = "C:\\Users\\Mult-e\\Desktop\\@Codigos\\MestradoCodigos\\MestradoCodigos\\Análises Resultados\\Classificação\\"
        
        if not baseline:
            csv_path = os.path.join(path_arquivos_csv_classificacao, f"{mecanismo}_{approach}", f"{imputer}{extension}")
        else:
            csv_path = os.path.join(path_arquivos_csv_classificacao, mecanismo, f"{extension}_baseline.csv")

        try:
            df = pd.read_csv(csv_path, index_col=0)
        except FileNotFoundError:
            self._logger.debug(f"Arquivo não encontrado: {csv_path}")
            

        f1 = 2 * (df['recall'] * df['precision']) / (df['recall'] + df['precision'])
        df['f1_score'] = round(f1, 3)

        try:
            df.to_csv(csv_path)
            self._logger.info(f"Resultados salvos com sucesso para o imputer: {imputer}")
        except Exception as e:
            self._logger.debug(f"Erro ao salvar resultados para o imputer {imputer}: {e}")

    # ------------------------------------------------------------------------
    def main(self,path:str, classifier:str):

        df_base = pd.read_csv(f'{path}/{classifier}_baseline.csv', index_col=0)
        df_knn = pd.read_csv(f'{path}/knn_{classifier}.csv', index_col=0)
        df_mean = pd.read_csv(f'{path}/mean_{classifier}.csv', index_col=0)
        df_mice = pd.read_csv(f'{path}/mice_{classifier}.csv', index_col=0)
        df_pmivae = pd.read_csv(f'{path}/pmivae_{classifier}.csv', index_col=0)
        df_saei = pd.read_csv(f'{path}/saei_{classifier}.csv', index_col=0)
        df_soft = pd.read_csv(f'{path}/softImpute_{classifier}.csv', index_col=0)
        df_gain = pd.read_csv(f'{path}/gain_{classifier}.csv', index_col=0)

        df_knn = ClassificationDatasets.calcula_dif(df_base,df_knn)
        df_mean = ClassificationDatasets.calcula_dif(df_base,df_mean)
        df_mice = ClassificationDatasets.calcula_dif(df_base,df_mice)
        df_pmivae = ClassificationDatasets.calcula_dif(df_base,df_pmivae)
        df_saei = ClassificationDatasets.calcula_dif(df_base,df_saei)
        df_soft = ClassificationDatasets.calcula_dif(df_base, df_soft)
        df_gain = ClassificationDatasets.calcula_dif(df_base, df_gain)

        df_knn = ClassificationDatasets.rename_col(df_knn, 'knn')
        df_mean = ClassificationDatasets.rename_col(df_mean, 'mean')
        df_mice = ClassificationDatasets.rename_col(df_mice, 'mice')
        df_pmivae = ClassificationDatasets.rename_col(df_pmivae, 'pmivae')
        df_saei = ClassificationDatasets.rename_col(df_saei, 'saei')
        df_soft = ClassificationDatasets.rename_col(df_soft, 'softImpute')
        df_gain = ClassificationDatasets.rename_col(df_gain, 'gain')

        df1 = df_mean[['dataset', 'missing_rate', 'mean_accuracy']].join(
            df_knn['knn_accuracy']
        )
        df1 = df1.join(df_mice['mice_accuracy'])
        df1 = df1.join(df_pmivae['pmivae_accuracy'])
        df1 = df1.join(df_saei['saei_accuracy'])
        df1 = df1.join(df_soft['softImpute_accuracy'])
        df1 = df1.join(df_gain['gain_accuracy'])

        df2 = df_mean[['dataset', 'missing_rate', 'mean_recall']].join(
            df_knn['knn_recall']
        )
        df2 = df2.join(df_mice['mice_recall'])
        df2 = df2.join(df_pmivae['pmivae_recall'])
        df2 = df2.join(df_saei['saei_recall'])
        df2 = df2.join(df_soft['softImpute_recall'])
        df2 = df2.join(df_gain['gain_recall'])

        df3 = df_mean[['dataset', 'missing_rate', 'mean_precision']].join(
            df_knn['knn_precision']
        )
        df3 = df3.join(df_mice['mice_precision'])
        df3 = df3.join(df_pmivae['pmivae_precision'])
        df3 = df3.join(df_saei['saei_precision'])
        df3 = df3.join(df_soft['softImpute_precision'])
        df3 = df3.join(df_gain['gain_precision'])

        df4 = df_mean[['dataset', 'missing_rate', 'mean_f1-score']].join(
            df_knn['knn_f1-score']
        )
        df4 = df4.join(df_mice['mice_f1-score'])
        df4 = df4.join(df_pmivae['pmivae_f1-score'])
        df4 = df4.join(df_saei['saei_f1-score'])
        df4 = df4.join(df_soft['softImpute_f1-score'])
        df4 = df4.join(df_gain['gain_f1-score'])

        output = f'{path}/Diferença dos algoritmos para baseline.xlsx'
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df1.to_excel(writer, sheet_name='Accuracy', index=False)
            df2.to_excel(writer, sheet_name='Recall', index=False)
            df3.to_excel(writer, sheet_name='Precision', index=False)
            df4.to_excel(writer, sheet_name='F1-score', index=False)