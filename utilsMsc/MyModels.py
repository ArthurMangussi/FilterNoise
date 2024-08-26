# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'

# Variational Autoencoder Filter for Bayesian Ridge Imputation
from Algoritmos.bridge import VAEBRIDGE
from Algoritmos.vae_bridge import ConfigVAE

# MICE, KNN, Dumb, missForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from Algoritmos.customknn import CustomKNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge

# Partial Multiple Imputation with Variational Autoencoders
from Algoritmos.pmivae import PMIVAE
from Algoritmos.vae_pmivae import ConfigVAE

# Siamese Autoencoder
from Algoritmos.saei import ConfigSAE, SAEImp, DataSets

# Soft impute 
from Algoritmos.soft_impute import SoftImpute

# Generative Adversarial Imputation Networks
from Algoritmos.gain import Gain

from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import warnings

from utilsMsc.MeLogSingle import MeLogger
from utilsMsc.MyUtils import MyPipeline

# Ignorar todos os avisos
warnings.filterwarnings("ignore")

class ModelsImputation:
    def __init__(self) :
        self._logger = MeLogger()
    # ------------------------------------------------------------------------
    @staticmethod
    def model_mice(dataset_train:pd.DataFrame):
        imputer = IterativeImputer(max_iter=100)
        mice = imputer.fit(dataset_train.iloc[:, :].values)

        return mice

    # ------------------------------------------------------------------------
    @staticmethod
    def model_knn(dataset_train:pd.DataFrame):
        imputer = KNNImputer(n_neighbors=5)
        knn = imputer.fit(dataset_train.iloc[:, :].values)

        return knn
    
    # ------------------------------------------------------------------------
    @staticmethod
    def model_knn_custom(dataset_train:pd.DataFrame, listTypes):
        imputer = CustomKNNImputer(n_neighbors=5, original=dataset_train, featuresType=listTypes)
        knn = imputer.fit(dataset_train.iloc[:, :].values)

        return knn

    # ------------------------------------------------------------------------
    @staticmethod
    def model_autoencoder_bridge(dataset_train, missing_feature_id, k_perc):
        vae_config = ConfigVAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.neurons = [15]
        vae_config.dropout_rates = [0.1]
        vae_config.latent_dimension = 5
        vae_config.number_features = dataset_train.shape[1]

        vae_bridge_model = VAEBRIDGE(
            vae_config, missing_feature_idx=missing_feature_id, k=k_perc
        )

        vae_bridge_model.fit(dataset_train)

        return vae_bridge_model

    # ------------------------------------------------------------------------
    @staticmethod
    def model_autoencoder_pmivae(dataset_train:pd.DataFrame, params:dict):
        original_shape = dataset_train.shape
        vae_config = ConfigVAE()
        vae_config.verbose = 0
        vae_config.batch_size = 128
        vae_config.validation_split=0.2
        vae_config.input_shape = (original_shape[1],)
        vae_config.epochs = params["n_epochs"]
        vae_config.latent_dimension = params["latent_dimension"]
        vae_config.neurons = params["neurons"]
        vae_config.dropout_fc = [0.3] * len(params["neurons"])

        pmivae_model = PMIVAE(vae_config, num_samples=200)
        model = pmivae_model.fit(dataset_train)

        return model

    # ------------------------------------------------------------------------
    @staticmethod
    def set_vae_config(X,neurons, latent_dimension, n_epochs):
        original_shape = X.shape
        vae_config = ConfigVAE()
        vae_config.verbose = 0
        vae_config.batch_size = 128
        vae_config.validation_split = 0.2
        vae_config.input_shape = (original_shape[1],)
        vae_config.neurons = neurons
        vae_config.latent_dimension = latent_dimension
        vae_config.epochs = n_epochs
        vae_config.dropout_fc = [0.3] * len(neurons)

        return vae_config

    # ------------------------------------------------------------------------
    @staticmethod
    def train_pmivae(config, X_train, X_test, X_test_complete):
        pmivae_model = PMIVAE(config, num_samples=200)
        pmivae_model.fit(X_train.iloc[:,:].values)
        output_test = pmivae_model.transform(X_test.iloc[:,:].values)
        mse = mean_squared_error(y_pred=output_test, y_true=X_test_complete)
        return mse

    # ------------------------------------------------------------------------
    @staticmethod
    def GridSearchPMIVAE(X_train, X_test, X_test_complete, param_grid):
        best_score = np.inf
        best_params = {}

        for n_epochs in param_grid["epochs"]:
            for n_latent_dimension in param_grid["latent_dimension"]:
                for nro_neurons in param_grid["neurons"]:
                    vae_config = ModelsImputation.set_vae_config(X_train, nro_neurons, n_latent_dimension, n_epochs)
                    mse = ModelsImputation.train_pmivae(vae_config, X_train, X_test,X_test_complete)
                    if mse < best_score:
                        best_score = mse
                        best_params = {
                            "n_epochs": n_epochs,
                            "latent_dimension": n_latent_dimension,
                            "neurons": nro_neurons
                        }
        return best_params, best_score
    
    
    # ------------------------------------------------------------------------
    @staticmethod
    def modelo_saei(
        dataset_train:pd.DataFrame,
        dataset_test:pd.DataFrame,
        dataset_train_md:pd.DataFrame,
        dataset_test_md:pd.DataFrame,
        input_shape,
    ):
        vae_config = ConfigSAE()
        vae_config.verbose = 0
        vae_config.epochs = 200
        vae_config.input_shape = (input_shape,)

        saei_model = SAEImp()
        prep = MyPipeline()
        x_train_pre = prep.pre_imputed_dataset(dataset_train_md)
        x_test_pre = prep.pre_imputed_dataset(dataset_test_md)

        dados = DataSets(
            x_train=dataset_train,
            x_val=dataset_test,
            x_train_md=dataset_train_md,
            x_val_md=dataset_test_md,
            x_train_pre=x_train_pre,
            x_val_pre=x_test_pre,
        )

        model = saei_model.fit(dados, vae_config)
        return model

    # ------------------------------------------------------------------------
    @staticmethod
    def model_dumb(dataset_train:pd.DataFrame):
        imputer = SimpleImputer(strategy="mean")
        dumb = imputer.fit(dataset_train.iloc[:, :].values)

        return dumb
    
    # ------------------------------------------------------------------------
    @staticmethod
    def model_softimpute(dataset_train:pd.DataFrame):
        imputer = SoftImpute()
        soft_impute = imputer.fit(dataset_train.iloc[:,:].values)
        return soft_impute

    # ------------------------------------------------------------------------
    @staticmethod
    def model_gain(dataset_train:pd.DataFrame):
        imputer = Gain()
        gain = imputer.fit(dataset_train.iloc[:,:].values)
        return gain
    
    # ------------------------------------------------------------------------
    @staticmethod
    def model_missForest(dataset_train:pd.DataFrame):
        rf = RandomForestRegressor(n_jobs=-1, 
                                   criterion="absolute_error", 
                                   n_estimators=10, 
                                   random_state=42)
        imputer = IterativeImputer(estimator=rf)
        missForest = imputer.fit(dataset_train.iloc[:, :].values)

        return missForest
    # ------------------------------------------------------------------------
    @staticmethod
    def model_bayesian(dataset_train:pd.DataFrame):
        br = BayesianRidge()
        imputer = IterativeImputer(estimator=br, max_iter=100)
        baye = imputer.fit(dataset_train.iloc[:, :].values)

        return baye

    
    # ------------------------------------------------------------------------
    def choose_model(self,model: str, x_train, **kwargs):
        match model:
            case "mice":
                self._logger.info("[MICE] Training...")
                return ModelsImputation.model_mice(x_train)

            case "knn":
                self._logger.info("[KNN] Training...")
                return ModelsImputation.model_knn(x_train)

            case "vaebridge":
                self._logger.info("[VAEBRIDGE] Training...")
                # Estratégia de pré-imputação, para os VAE é utilizando a média
                X_treino_pre_imput = x_train.fillna(
                    np.mean(x_train[kwargs["col_name"]])
                )

                return ModelsImputation.model_autoencoder_bridge(
                    X_treino_pre_imput.loc[:, :].values,
                    kwargs["missing_feature_id"],
                    kwargs["k_perc"],
                )

            case "pmivae":
                self._logger.info("[PMIVAE] GridSearch...")
                params = {
                        "epochs": [200],
                        "latent_dimension": [5,10],
                        "neurons": [[np.shape(x_train)[0]/2],
                                    [np.shape(x_train)[0]/2, np.shape(x_train)[0]/4]],
                    }
                best_params, best_score = ModelsImputation.GridSearchPMIVAE(X_train=x_train,
                                                                            X_test=kwargs["x_test"],
                                                                            param_grid=params,
                                                                            X_test_complete=kwargs["x_test_complete"])
                
                self._logger.info(f"Best params for PMIVAE: {best_params}")
                self._logger.info(f"Best score found in GridSearch (MSE): {best_score}")
                
                return ModelsImputation.model_autoencoder_pmivae(x_train.loc[:, :].values, 
                                                                 params=best_params)

            case "saei":
                self._logger.info("[SAEI] Training...")
                return ModelsImputation.modelo_saei(
                    dataset_train=x_train,
                    dataset_test=kwargs["x_test"],
                    dataset_train_md=kwargs["x_train_md"],
                    dataset_test_md=kwargs["x_test_md"],
                    input_shape=kwargs["input_shape"],
                )

            case "mean":
                self._logger.info("[MEAN] Training...")
                return ModelsImputation.model_dumb(x_train)
            
            case "softImpute":
                self._logger.info("[SoftImpute] Training...")
                return ModelsImputation.model_softimpute(x_train)
            
            case "gain":
                self._logger.info("[GAIN] Training...")
                return ModelsImputation.model_gain(x_train)
            
            case "missForest":
                self._logger.info("[missForest] Training...")
                return ModelsImputation.model_missForest(x_train)
            case "customKNN":
                self._logger.info("[KNN-HEOM] Training...")
                return ModelsImputation.model_knn_custom(dataset_train=x_train,
                                                         listTypes=kwargs['listTypes'])
            case "bayesian":
                self._logger.info("[Bayesian Ridge] Training...")
                return ModelsImputation.model_bayesian(dataset_train=x_train)
