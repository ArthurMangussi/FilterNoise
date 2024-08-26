import sys
sys.path.append("./")

from utilsMsc.MyUtils import MyPipeline
from utilsMsc.MyResults import AnalysisResults
from utilsMsc.MyPreprocessing import PreprocessingDatasets
from utilsMsc.MyModels import ModelsImputation

from sklearn.model_selection import StratifiedKFold
from time import perf_counter
import warnings
import os
import pandas as pd
from mdatagen.multivariate.mMNAR import mMNAR
from imblearn.under_sampling import EditedNearestNeighbours
import numpy as np 

from utilsMsc.MeLogSingle import MeLogger
from MyNoise import Noise

# Ignorar todos os avisos
warnings.filterwarnings("ignore")

def cria_tabela_sinteticos(path:str):

    datasets = MyPipeline.carrega_datasets(path)

    arquivos=os.listdir(path)
    nomes_sem_extensao = [os.path.splitext(arquivo)[0] for arquivo in arquivos]

    tabela_resultados ={}
    tabela_resultados["missing_rate"] = [5, 10, 20]
    tabela_resultados["nome_datasets"] = nomes_sem_extensao

    tabela_resultados["datasets"] = []
    for n_data in nomes_sem_extensao:
        tabela_resultados["datasets"].append(datasets[n_data])

    return tabela_resultados


def init_datasets(flag:bool, mecanismo:str):
    if flag:
        diretorio_synthetic = "C:\\Users\\Mult-e\\Desktop\\@Codigos\\Datasets sintéticos"
        tabela_resultados = cria_tabela_sinteticos(diretorio_synthetic)
        
    else:
        diretorio_principal = "C:\\Users\\Mult-e\\Desktop\\@Codigos\\MestradoCodigos\\MestradoCodigos\\Noise\\DatasetsNoise"
        datasets = MyPipeline.carrega_datasets(diretorio_principal)
        noise = Noise(datasets)
        tabela_resultados = noise.cria_tabela()
        
    # Cria diretórios para salvar os resultados do experimento
    os.makedirs(f"./Noise/Tempos/{mecanismo}_Multivariado", exist_ok=True)
    os.makedirs(f"./Noise/Datasets/{mecanismo}_Multivariado", exist_ok=True)
    os.makedirs(f"./Noise/Resultados/{mecanismo}_Multivariado", exist_ok=True)
    return tabela_resultados


def pipeline_noise(model_impt:str, mecanismo:str, tabela_resultados:dict):
    _logger = MeLogger()
    n_metricas = 1
    inicio = perf_counter()
    cv = StratifiedKFold(n_splits=5)
    enn = EditedNearestNeighbours(n_neighbors=5, kind_sel = "mode", sampling_strategy="majority")

    with open(f"./Noise/Tempos/{mecanismo}_Multivariado/tempo_{model_impt}.txt", "w") as file:
        # Gerando resultados para os mecanismo
        for dados, nome in zip(tabela_resultados["datasets"], tabela_resultados["nome_datasets"]):
            for md in tabela_resultados["missing_rate"]:
                df = dados.copy()
                file.write(f"\nDataset = {nome} com MD = {md}\n")
                fold = 0

                X = df.drop(columns = 'target')
                y = df['target'].values
                x_cv = X.values

                # Cross-validation with 5 folds
                for train_index, test_index in cv.split(x_cv, y):
                    _logger.info(f"Dataset = {nome} com MD = {md} imputed by = {model_impt} no Fold = {fold}")
                    x_treino, x_teste = x_cv[train_index], x_cv[test_index]
                    y_treino, y_teste = y[train_index], y[test_index]

                    X_treino = pd.DataFrame(x_treino, columns=X.columns)                    
                    X_teste = pd.DataFrame(x_teste, columns=X.columns) 
                    X_treino["target"] = y_treino
                    X_teste["target"] = y_teste                   

                    # Inicializando o normalizador (scaler)
                    scaler = PreprocessingDatasets.inicializa_normalizacao(X_treino)

                    # Normalizando os dados
                    X_treino_norm = PreprocessingDatasets.normaliza_dados(scaler, X_treino)
                    X_teste_norm = PreprocessingDatasets.normaliza_dados(scaler, X_teste)

                    # Subset para gerar os missing separadamente
                    X_treino_one = X_treino_norm[X_treino_norm.target == 1].reset_index(drop=True)
                    X_treino_zero = X_treino_norm[X_treino_norm.target == 0].reset_index(drop=True)
                    
                    # Geração dos missing values em cada conjunto de forma independente
                    ## Classe positiva    
                    y_treino_one = X_treino_one["target"].values                
                    generator_train_one = mMNAR(X=X_treino_one.drop(columns="target"), y=y_treino_one)
                    df_md_one = generator_train_one.random(missing_rate=md/2, deterministic=False)
                    df_md_one = df_md_one.drop(columns="target")
                    df_md_one["target"] = y_treino_one
                    
                    ## Classe negativa
                    y_treino_zero = X_treino_zero["target"].values
                    generator_train_zero = mMNAR(X=X_treino_zero.drop(columns="target"), y=y_treino_zero)
                    df_md_zero = generator_train_zero.random(missing_rate=md/2, deterministic=False)
                    df_md_zero = df_md_zero.drop(columns="target")
                    df_md_zero["target"] = y_treino_zero

                    ## Conjunto de teste
                    generator_test = mMNAR(X=X_teste_norm, y = y_teste)
                    X_teste_norm_md = generator_test.random(missing_rate=md, deterministic=False)
                    X_teste_norm_md = X_teste_norm_md.drop(columns="target")
                    
                    # Salvando as linhas missing
                    linhas_nan_one = df_md_one[np.isnan(df_md_one).any(axis=1)].drop(columns="target")
                    linhas_nan_zero = df_md_zero[np.isnan(df_md_zero).any(axis=1)].drop(columns="target")

                    # Dropando os missing 
                    df_md_one = df_md_one.dropna()
                    df_md_zero = df_md_zero.dropna()

                    # Filtrando os dados remasnecentes
                    unified_dataset = pd.concat([df_md_one, df_md_zero])

                    unified_y = unified_dataset["target"].values
                    unified_dataset = unified_dataset.drop(columns="target")

                    X_train_filtered, y_train_filtered = enn.fit_resample(unified_dataset, unified_y)

                    # Concatenando o dataset filtrado e as linhas missing
                    filtered_dataset = X_train_filtered.copy()

                    filtered_md_dataset = pd.concat([filtered_dataset, linhas_nan_one, linhas_nan_zero])
                    filtered_md_dataset_train = filtered_md_dataset.sample(frac=1).reset_index(drop=True)
                            
                    inicio_imputation = perf_counter()

                    # Inicializando e treinando o modelo
                    model_selected = ModelsImputation()
                    if model_impt == "saei":
                        # SAEI
                        features = filtered_md_dataset_train.columns[filtered_md_dataset_train.isna().any()].tolist()
                        model = model_selected.choose_model(model = model_impt, 
                                                        x_train = filtered_dataset, 
                                                        x_test = X_teste_norm.drop(columns="target"),
                                                        x_train_md = filtered_md_dataset_train,
                                                        x_test_md = X_teste_norm_md,
                                                        col_name =  features,
                                                        input_shape = filtered_dataset.shape[1])
                        
                    # KNN, MICE, PMIVAE, MEAN, SOFT IMPUTE, GAIN
                    else:
                        model = model_selected.choose_model(
                            model=model_impt,
                            x_train=filtered_md_dataset_train,
                            y_train = y_train_filtered,
                            x_val = X_teste_norm_md,
                            y_val = y_teste)


                    fim_imputation = perf_counter()
                    file.write(f'Tempo de treinamento para fold = {fold} foi = {fim_imputation-inicio_imputation:.4f} s\n')

                    # Imputação dos missing values nos conjuntos de treino e teste
                    try:
                        output_md_test = model.transform(
                            X_teste_norm_md.iloc[:, :].values
                        )
                    except AttributeError:                        
                        fatores_latentes_test = model.fit(X_teste_norm_md.iloc[:, :].values)
                        output_md_test = model.predict(X_teste_norm_md.iloc[:, :].values)

                    # Calculando MAE para a imputação no conjunto de teste
                    (
                        mae_teste_mean,
                        mae_teste_std,
                    ) = AnalysisResults.gera_resultado_multiva(
                        resposta=output_md_test,
                        dataset_normalizado_md=X_teste_norm_md,
                        dataset_normalizado_original=X_teste_norm,
                    )

                    tabela_resultados[
                        f'{model_impt}/{nome}/{md}/{fold}/MAE'
                    ] = {'teste': round(mae_teste_mean,3)}

                    # Dataset imputado
                    data_imputed = pd.DataFrame(output_md_test.copy(), columns=X.columns)
                    data_imputed['target'] = y_teste

                    data_imputed.to_csv(f"./Noise/Datasets/{mecanismo}_Multivariado/{nome}_{model_impt}_fold{fold}_md{md}.csv", index=False)
                    fold += 1

            resultados_final = AnalysisResults.extrai_resultados(tabela_resultados)

            # Resultados da imputação
            resultados_mecanismo = AnalysisResults.calcula_metricas_estatisticas_resultados(
                resultados_final, n_metricas, fold
            )

        resultados_mecanismo.to_csv(f"./Noise/Resultados/{mecanismo}_Multivariado/{model_impt}_Filtered.csv")

        fim = perf_counter()
        file.write(f'Tempo de total de processamento total para {model_impt.upper()} foi = {fim-inicio:.4f} s')


if __name__ == "__main__":
    # Datasets
    synthetic = True
    #mecanismo = "MNAR-determisticFalse"
    mecanismo_syn = "MNAR-determisticFalse-SyntheticFiltered"
    
    tabela_resultados = init_datasets(synthetic, mecanismo_syn)

    for model_impt in [#"mean",
                       #"softImpute",
                       #"bayesian",
                       #"knn",
                       #"mice",
                       #"gain",
                       #"pmivae",
                       "missForest"
                       ]:
        pipeline_noise(model_impt, mecanismo_syn, tabela_resultados)
