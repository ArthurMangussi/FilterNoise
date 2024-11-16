import sys 
sys.path.append("./")
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from utilsMsc.MyPreprocessing import PreprocessingDatasets
from utilsMsc.MeLogSingle import MeLogger

def pipeline_classifica_noise(nome_dataset:str,
                              model_impt:str, 
                              md:str, 
                              df:pd.DataFrame,
                              parameters:dict):

    _logger = MeLogger()
    _logger.info(f"Dataset = {nome_dataset} com MD = {md} imputed by = {model_impt}")

    cv = StratifiedKFold()
    model = RandomForestClassifier(random_state=42,n_jobs=-1)
    X = df.drop(columns = 'target')
    y = df['target'].values
    x_cv = X.values

    all_f1 = {}
    fold = 0
    for train_index, test_index in cv.split(x_cv, y):
        x_treino, x_teste = x_cv[train_index], x_cv[test_index]
        y_treino, y_teste = y[train_index], y[test_index]

        X_treino = pd.DataFrame(x_treino, columns=X.columns)                    
        X_teste = pd.DataFrame(x_teste, columns=X.columns) 

        # Inicializando o normalizador (scaler)
        scaler = PreprocessingDatasets.inicializa_normalizacao(X_treino)

        # Normalizando os dados
        X_treino_norm = PreprocessingDatasets.normaliza_dados(scaler, X_treino)
        X_teste_norm = PreprocessingDatasets.normaliza_dados(scaler, X_teste)

        clf = RandomizedSearchCV(estimator=model,
                                 param_distributions=parameters,
                                 cv=5,
                                 random_state=42,
                                 verbose=False)
        clf.fit(X_treino_norm, y_treino)

        best_model = clf.best_estimator_

        y_pred = best_model.predict(X_teste_norm)

        f1 = f1_score(y_true=y_teste, y_pred=y_pred)

        all_f1[f"{name_dataset}_{model_impt}_md{md}_fold{fold}"] = f1
        fold += 1

    return all_f1

if __name__ == "__main__":
    path = "./Imputed Datasets/MNAR-determisticFalse_Multivariado/All"

    tabela_final_resultados = {}
    parameters = {"n_estimators": np.arange(10, 100, 10),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2),
           "max_features": np.arange(0.0,1.0,0.1)}
    
    for model_impt in ["mean",
                       "softImpute",
                       "bayesian",
                       "knn",
                       "mice",
                       "gain",
                       "pmivae",
                       "missForest"]:
        
        for name_dataset in ["wiscosin",
                            "pima",                                            
                            "indian_liver",
                            "parkinsons",
                            "mammographic_masses",                                            
                            "thoracic_surgery",                                            
                            "diabetic_retionapaty",
                            "thyroid_recurrence",
                            "blood_transfusion",
                            "law"]:
            
            for md in [5,10,20]:
                complete_path = f"{path}/{name_dataset}_{model_impt}_md{md}.csv"

                df = pd.read_csv(complete_path)

                resultados_f1 = pipeline_classifica_noise(name_dataset,
                                        model_impt,
                                        md,
                                        df,
                                        parameters
                                        )
                media_f1 = np.mean([values for keys, values in resultados_f1.items()])
                std = np.std([values for keys, values in resultados_f1.items()])
                tabela_final_resultados[f"{name_dataset}_{model_impt}_md{md}"] = f"{round(media_f1,3)}-{round(std,3)}"
                
        resultados = pd.DataFrame([tabela_final_resultados]).T
        resultados.to_csv(f"./Resultados/f1_mnar_{model_impt}.csv")