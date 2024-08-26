# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

__author__ = 'Arthur Dantas Mangussi'

from utilsMsc.pycol import Complexity

import numpy as np
import pandas as pd
from openpyxl import load_workbook
import arff
import os 

class ComplexityDatasets:
    def __init__(self, baseline_dataset) -> None:
        self.base = baseline_dataset
    # ------------------------------------------------------------------------
    @staticmethod
    def analisa_complexidade(path:str)->dict:
        complexity = Complexity(path)
        return {
            # Feature Overlap
            "f1": complexity.F1(),
            "f1v": complexity.F1v(),
            "f2": complexity.F2(),
            "f3": complexity.F3(),
            "f4": complexity.F4(),
            "in": complexity.input_noise(),
            # Instance Overlap 
            "r-value": complexity.R_value(),
            "degOver": complexity.deg_overlap(),
            "n3": complexity.N3(),
            "si": complexity.SI(),
            "n4": complexity.N4(),
            "kdn": complexity.kDN(),
            "d3": complexity.D3_value(),
            "cm": complexity.CM(),
            "borderline": complexity.borderline(),
            # Structural Overlap
            "n1": complexity.N1(),
            "t1": complexity.T1(),
            "clst": complexity.Clust(),
            # "onb": complexity.ONB_avg(),
            "lscavg": complexity.LSC(),
            "dbc": complexity.DBC(),
            "n2": complexity.N2(),
            "nsg": complexity.NSG(),
            "icsv": complexity.ICSV()
        }
    
    # ------------------------------------------------------------------------
    def diff(self, nome, df):
        # Implementar as outras métricas de complexidade
        F1 = self.base[nome][0] - df['f1_mean']
        L1 = self.base[nome][1] - df['l1_mean']
        N1 = self.base[nome][3] - df['n1_mean']
        N2 = self.base[nome][4] - df['n2_mean']
        N3 = self.base[nome][5] - df['n3_mean']

        return F1.to_list(), L1.to_list(), N1.to_list(), N2.to_list(), N3.to_list()
    
    # ------------------------------------------------------------------------
    def tabela_complexidade_diferenca(self,dataframe):

        results = {}

        results['imputer'] = dataframe['imputer'].to_list()
        results['Dataset'] = dataframe['dataset'].to_list()
        results['Missing Rate'] = dataframe['missing rate'].to_list()

        results.update({'F1': [], 'L1': [], 'N1': [], 'N2': [], 'N3': []})

        datasets = [
            "acute_inflammationsUrinary",
            "acute_inflammationsNephritis",
            "autism_adult",
            "autism_adolescent",
            "autism_child",
            "blood_transfusion",
            "diabetic_retinopathy",
            "thyroid_recurrence",
            "fertility",
            "haberman",
            "immunotherapy",
            "maternal_healthRisk",
            "npha_doctor",
            "sa_heart",
            "vertebral_column",
            "wiscosin",
            "pima_diabetes",
            "bc_coimbra",
            "indian_liver",
            "parkinsons",
            "mammographic_masses",
            "hcv_egyptian",
            "thoracic_surgey",
            "glioma",
            "cirrhosis",
            "iris",
            "wine",
            "heart_cleveland",
            "german_credit",
            "bc_yugoslavia",
            "student_dropout",
            "glass_identification",
            "credit_approval",
            "thyrroid_disease",
            "obesity_eating",
            "heart_failure",
            "hepatitis",
            "tictac_toe",
            "balance_scale",
            "dermatology",
            "ecoli",
            "early_diabetes",
            "contraceptive_method",
            "higher_education",
            "MONK_problem1",
            "MONK_problem2",
            "MONK_problem3",
            "land_mines",
            "echocardiogram",
            "rice",
            "mushroom",
            "nursery",
            "darwin",
            "auction_verification",
            "caesarin",
            "cryotherapy",
            "sirtuin6",
            "sani",
            "primary_tumor",
            "myocardial",
            "lymphography",
            "polish_companies1",
            "polish_companies2",
            "polish_companies3",
            "polish_companies4",
            "polish_companies5",
            "cervical_center",
            "cardiovascular",
            "audiology",
            "post_operative",
            "bone_marrow",
            "user_knowledge",
            "student_performance",
            "teach_assistant",
            "breast_tissue",
            "divorce_predictors",
            "qsar_biodregad",
            "turkish_music",
            "steel_plates",
            "krvVsKp",
            "phoneme",
            "eeg_eye",
            "bank_marketing",
            "abalone",
            "analcatdata",
            "proba_football",
            "car",
            "yeast",
            "kropt",
            "higgs_df"
        ]

        passo = 4

        for dataset in datasets:
            F1, L1, N1, N2, N3 = self.diff(
                dataset,
                dataframe[
                    datasets.index(dataset)
                    * passo : datasets.index(dataset)
                    * passo
                    + passo
                ],
            )

            for VF1, VL1, VN1, VN2, VN3 in zip(F1, L1, N1, N2, N3):
                results['F1'].append(VF1)
                results['L1'].append(VL1)
                results['N1'].append(VN1)
                results['N2'].append(VN2)
                results['N3'].append(VN3)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------------
    @staticmethod
    def cria_arquivo_arff(mecanismo:str,tabela_resultados):
        
        for dados, nome in zip(
            tabela_resultados['datasets'], tabela_resultados['nome_datasets']
        ):

            target_values = dados.target
            dados = dados.drop(columns='target')
            dados['target'] = target_values
            
            arff_content = ComplexityDatasets.save_arff(data_complex=dados,
                                                        mechanism=mecanismo,
                                                        flag=False)

            # Salvar o conteúdo ARFF em um arquivo
            with open(
                f'./Análises Resultados/Complexidade/{mecanismo}/baseline/{nome}.arff', 'w'
            ) as fcom:
                fcom.write(arff_content)

    # ------------------------------------------------------------------------
    @staticmethod
    def tabela_complexidade(mecanismo:str, imputer: str):

        wb = load_workbook(
            f'Análises Resultados/Complexidade/{mecanismo}/{imputer}_complexity.xlsx'
        )

        results = {
            'imputer': [],
            'dataset': [],
            'f1_mean': [],
            'f1_std': [],
            'l1_mean': [],
            'l1_std': [],
            'n1_mean': [],
            'n1_std': [],
            'n2_mean': [],
            'n2_std': [],
            'n3_mean': [],
            'n3_std': [],
        }

        for sheet in wb:
            dataset = sheet.title
            results['dataset'].append(dataset)
            results['imputer'].append(imputer)

            # Lista para armazenar os valores das células
            valores_f1 = []
            valores_l1 = []
            valores_n1 = []
            valores_n2 = []
            valores_n3 = []

            # Colunas de interesse
            colunas_interesse = ['B', 'C', 'D', 'E', 'F']

            # Iterar sobre as colunas
            for coluna in colunas_interesse:
                # Construir a coordenada da célula (ex: B2, C2, ...)
                f1 = f'{coluna}2'
                l1 = f'{coluna}3'
                n1 = f'{coluna}5'
                n2 = f'{coluna}6'
                n3 = f'{coluna}7'

                valor_f1 = sheet[f1].value
                valores_f1.append(valor_f1)
                valor_l1 = sheet[l1].value
                valores_l1.append(valor_l1)
                valor_n1 = sheet[n1].value
                valores_n1.append(valor_n1)
                valor_n2 = sheet[n2].value
                valores_n2.append(valor_n2)
                valor_n3 = sheet[n3].value
                valores_n3.append(valor_n3)
            try:
                f1_mean, f1_std = np.mean(valores_f1), np.std(valores_f1)
                l1_mean, l1_std = np.mean(valores_l1), np.std(valores_l1)
                n1_mean, n1_std = np.mean(valores_n1), np.std(valores_n1)
                n2_mean, n2_std = np.mean(valores_n2), np.std(valores_n2)
                n3_mean, n3_std = np.mean(valores_n3), np.std(valores_n3)
            except TypeError:
                f1_mean, f1_std = 0, 0
                l1_mean, l1_std = 0, 0
                n1_mean, n1_std = 0, 0
                n2_mean, n2_std = 0, 0
                n3_mean, n3_std = 0, 0


            results['f1_mean'].append(f1_mean)
            results['f1_std'].append(f1_std)

            results['l1_mean'].append(l1_mean)
            results['l1_std'].append(l1_std)

            results['n1_mean'].append(n1_mean)
            results['n1_std'].append(n1_std)

            results['n2_mean'].append(n2_mean)
            results['n2_std'].append(n2_std)

            results['n3_mean'].append(n3_mean)
            results['n3_std'].append(n3_std)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------------
    @staticmethod
    def save_arff(data_complex, mechanism, flag=True,**kwargs):
        
        if flag:
            approach, model_impt, nome = kwargs['approach'], kwargs['model_impt'], kwargs['name']

            os.makedirs(
                f'./Análises Resultados/Complexidade/{mechanism}_{approach}/{model_impt}/',
                exist_ok=True,
            )
        atts = ComplexityDatasets.formata_arff(data_complex, nome)

        dictarff = {
                    'attributes': atts,
                    'data': data_complex.values.tolist(),
                    'relation': f'{nome}',
                    }

        # Criar o arquivo ARFF
        return arff.dumps(dictarff)
    
    # ------------------------------------------------------------------------
    @staticmethod
    def formata_arff(data_imputed_complete, name):
        attributes = []
        for j in data_imputed_complete:
            if (
                data_imputed_complete[j].dtypes
                in ['int64', 'float64', 'float32']
                and j != 'target'
            ):
                attributes.append((j, 'NUMERIC'))
            elif j == 'target':
                if (name == 'bc_coimbra' 
                    or name == 'indian_liver'
                    or name == "blood_transfusion"
                    or name == "haberman"
                    or name == "sa_heart") :
                    attributes.append((j, ['1.0', '2.0']))
                elif (name == 'hcv_egyptian'
                        or name == "npha_doctor"
                        or name == "vertebral_column"
                        or name == "maternal_healthRisk"
                        or name == "cirrhosis"
                        or name == "iris"
                        or name == "wine"
                        or name == "student_dropout"
                        or name == "glass_identification"
                        or name == "obesity_eating"
                        or name == "balance_scale"
                        or name == "dermatology"
                        or name == "ecoli"
                        or name == "contraceptive_method"
                        or name == "higher_education"
                        or name == "land_mines"
                        or name == "nursery"
                        or name == "primary_tumor"
                        or name == "lymphography"
                        or name == "audiology"
                        or name == "post_operative"
                        or name == "user_knowledge"
                        or name == "student_performance"
                        or name == "teach_assistant"
                        or name == "turkish_music"
                        or name == "abalone"
                        or name == "analcatdata"
                        or name == "car"
                        or name == "yeast"
                        or name == "kropt"):
                    attributes.append(
                        (
                            j,
                            sorted(data_imputed_complete[j]
                            .unique()
                            .astype(str)
                            )
                        )
                    )
                else:
                    attributes.append((j, ['1.0', '0.0']))
            else:
                attributes.append(
                    (
                        j,
                        sorted(data_imputed_complete[j]
                        .unique()
                        .astype(str)
                        )
                    )
                )

        return attributes