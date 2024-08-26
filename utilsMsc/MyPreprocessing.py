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

    # ------------------------------------------------------------------------
    def pre_processing_datasets(self, datasets: dict):
        acute_inflammations = datasets["diagnosis1"].copy()
        acute_inflammations = acute_inflammations.drop(columns="target.1")
        acute_inflammations = self.label_encoder(acute_inflammations, ["Nausea", 
                                                                        "Lumbar",
                                                                        "Urine",
                                                                        "Micturition",
                                                                        "Burning",
                                                                        "target"])

        acute_nephritis = datasets["diagnosis1"].copy()
        acute_nephritis = acute_nephritis.drop(columns="target")
        acute_nephritis = self.label_encoder(acute_nephritis, ["Nausea", 
                                                                "Lumbar",
                                                                "Urine",
                                                                "Micturition",
                                                                "Burning",
                                                                "target.1"])
        acute_nephritis = acute_nephritis.rename(columns={"target.1":"target"})

        autism_df = datasets["Autism-Adult-Data"].copy()
        autism_df = autism_df.drop(columns=["age_desc", "ethnicity", "contry_of_res", "relation"],
                                index = 52).dropna().reset_index(drop=True)
        autism_df = self.label_encoder(autism_df, ["gender",
                                                        "jundice",
                                                        "austim",
                                                        "used_app_before",
                                                        "target"])
        
        autism_child_df = datasets["Autism-Child-Data"].copy()
        autism_child_df = autism_child_df.drop(columns=["age_desc", "ethnicity", "contry_of_res", "relation"]).dropna().reset_index(drop=True)
        autism_child_df = self.label_encoder(autism_child_df, ["gender",
                                                        "jundice",
                                                        "austim",
                                                        "used_app_before",
                                                        "target"])

        autism_adoles_df = datasets["Autism-Adolescent-Data"].copy()
        autism_adoles_df = autism_adoles_df.drop(columns=["age_desc", "ethnicity", "contry_of_res", "relation"]).dropna().reset_index(drop=True)
        autism_adoles_df = self.label_encoder(autism_adoles_df, ["gender",
                                                        "jundice",
                                                        "austim",
                                                        "used_app_before",
                                                        "target"])

        blood_df = datasets["blood-transfusion-service-center"].copy()
        blood_df["target"] = blood_df["target"].astype("int64")

        messidor_df = datasets["messidor_features"].copy()
        messidor_df["target"] = messidor_df["target"].astype("int64")
        
        thyroid_df = datasets["Thyroid_Diff"].copy()
        thyroid_df = self.label_encoder(thyroid_df, ["Gender", 
                                                   "Smoking", 
                                                   "Hx Smoking",
                                                   "Hx Radiothreapy",
                                                   "Focality",
                                                   "target"
                                                   ])

        thyroid_df = self.ordinal_encoder(thyroid_df, ["Physical Examination",
                                                        "Thyroid Function",
                                                        "Pathology",
                                                        "Risk",
                                                        "T",
                                                        "N",
                                                        "M",
                                                        "Stage",
                                                        "Response"])

        thyroid_df = self.one_hot_encode(thyroid_df, ["Adenopathy"])

        fertility_df = datasets["fertility_Diagnosis"].copy()
        map_season = {-1.0:"winter",
               -0.33:"spring",
               0.33:"summer",
               -1.0:"fall"}

        map_fever = {-1.0: "less than three months ago",
                    0.0: "more than three months ago",
                    1.0: "no"}

        map_smoking = {-1.0:"never",
                    0.0:"occasional",
                    1.0: "daily"}

        fertility_df["Season"] = fertility_df["Season"].map(map_season)
        fertility_df[' high fevers'] = fertility_df[' high fevers'].map(map_fever)
        fertility_df["smoking"] = fertility_df["smoking"].map(map_smoking)

        fertility_df = self.label_encoder(fertility_df, ["target"])

        fertility_df = self.one_hot_encode(fertility_df, ["Season"])

        fertility_df = self.ordinal_encoder(fertility_df, [' high fevers',
                                                                " alcohol consumption",
                                                                "smoking"])

        haberman_df = datasets["dataset_43_haberman"].copy()
        haberman_df["target"] = haberman_df["target"].astype("int64")

        immunotherapy_df = datasets["Immunotherapy"].copy()
        immunotherapy_df = self.ordinal_encoder(immunotherapy_df, ["Type"])
        immunotherapy_df = self.label_encoder(immunotherapy_df, ["sex"])

        maternal_health_risk_df = datasets["Maternal Health Risk Data Set"].copy()
        maternal_health_risk_df = self.label_encoder(maternal_health_risk_df, ["target"])

        npha_doctor_visits_df = datasets["NPHA-doctor-visits"].copy()
        npha_doctor_visits_df["target"] = npha_doctor_visits_df["target"].astype("int64")

        sa_heart = datasets["sa-heart"].copy()
        sa_heart["target"] = sa_heart["target"].astype("int64")
        
        vertebral_df = datasets["vertebral-column"].copy()
        vertebral_df["target"] = vertebral_df["target"].astype("int64")

        breast_cancer_wisconsin_df = datasets["wiscosin"].copy()
        breast_cancer_wisconsin_df = breast_cancer_wisconsin_df.drop(columns="ID")
        breast_cancer_wisconsin_df = self.label_encoder(
            breast_cancer_wisconsin_df, ["target"]
        )

        pima_diabetes_df = datasets["pima_diabetes"].copy()

        bc_coimbra_df = datasets["bc_coimbra"].copy()

        indian_liver_df = datasets["indian_liver"].copy()
        indian_liver_df = indian_liver_df.dropna()
        indian_liver_df = self.label_encoder(indian_liver_df, ["Gender"])

        datasets["parkinsons"] = datasets["parkinsons"].drop(columns="name")
        parkinsons_df = datasets["parkinsons"].copy()

        mammographic_masses_df = datasets["mammographic_masses"].copy()
        mammographic_masses_df = (
            datasets["mammographic_masses"]
            .replace("?", np.nan)
            .dropna()
            .drop(columns="BI-RADS assessment")
            .reset_index(drop=True)
        )
        mammographic_masses_df["Age"] = mammographic_masses_df["Age"].astype("int64")
        mammographic_masses_df["Density"] = mammographic_masses_df["Density"].astype(
            "int64"
        )
        mammographic_masses_df = self.one_hot_encode(
            mammographic_masses_df, ["Shape", "Margin"]
        )

        hcv_egyptian_df = datasets["HCV-Egy-Data"].copy()

        thoracic_surgery_df = datasets["ThoraricSurgery"].copy()
        thoracic_surgery_df = self.label_encoder(
            self.one_hot_encode(thoracic_surgery_df, ["DGN"]),
            [
                "PRE7",
                "PRE8",
                "PRE9",
                "PRE10",
                "PRE11",
                "PRE17",
                "PRE19",
                "PRE25",
                "PRE30",
                "PRE32",
                "target",
            ],
        )

        thoracic_surgery_df = self.ordinal_encoder(thoracic_surgery_df, ["PRE6","PRE14"])
        
        glioma = datasets["TCGA_InfoWithGrade"].copy()

        cirrhosis_df = datasets['cirrhosis'].copy()
        cirrhosis_df = cirrhosis_df.dropna().drop(columns="ID")
        cirrhosis_df = self.label_encoder(cirrhosis_df, ["target",
                                                         "Sex",
                                                         "Ascites",
                                                         "Hepatomegaly",
                                                         "Spiders",
                                                         "Edema"])
        cirrhosis_df = self.ordinal_encoder(cirrhosis_df, ["Stage"])
        cirrhosis_df = self.one_hot_encode(cirrhosis_df, ["Drug"]).dropna()

        iris_df = datasets['iris'].copy()
        iris_df = self.label_encoder(iris_df, ["target"])

        wine_df = datasets['wine'].copy()

        heart_cleveland = datasets['cleveland'].copy()
        heart_cleveland = heart_cleveland["target"].astype("float64")

        german_credit_df = datasets['german'].copy()
        german_credit_df = self.label_encoder(german_credit_df, ["Att18",
                                                                 "Att19"])
        german_credit_df = self.ordinal_encoder(german_credit_df, ["Att0",
                                                                   "Att5",
                                                                   "Att6",
                                                                   "Att11"])
        german_credit_df = self.one_hot_encode(german_credit_df, ["Att2",
                                                                  "Att3",
                                                                  "Att8",
                                                                  "Att9",
                                                                  "Att13",
                                                                  "Att14",
                                                                  "Att16"])

        bc_yugoslavia_df = datasets['breast-cancer-yugo'].copy()
        bc_yugoslavia_df = bc_yugoslavia_df.replace("?", np.nan).dropna()
        bc_yugoslavia_df = self.label_encoder(bc_yugoslavia_df, ["target",
                                                                 "node-caps",
                                                                 "breast",
                                                                 "irradiat"])
        bc_yugoslavia_df = self.ordinal_encoder(bc_yugoslavia_df, ["age",
                                                                   "tumor-size",
                                                                   "inv-nodes",
                                                                   "deg-malig",
                                                                   "menopause"])
        bc_yugoslavia_df = self.one_hot_encode(bc_yugoslavia_df, ["breast-quad"]).dropna()
        
        student_dropout = datasets['predict_students'].copy()
        student_dropout = self.label_encoder(student_dropout, ["target"])

        glass_identification = datasets['glass'].copy()
        glass_identification = self.label_encoder(glass_identification, ["target"]).drop(columns="ID")

        credit_approval_df = datasets['crx'].copy()
        credit_approval_df = credit_approval_df.replace("?", np.nan).dropna()
        credit_approval_df = self.label_encoder(credit_approval_df, ["target",
                                                                     "A1",
                                                                     "A9",
                                                                     "A10",
                                                                     "A12",
                                                                     "A4",
                                                                     "A5",
                                                                     "A6",
                                                                     "A7",
                                                                     "A13"])

        thyrroid_disease = datasets['thyroid'].copy()

        obesity_eating = datasets['ObesityDataSet_raw_and_data_sinthetic'].copy()
        obesity_eating = self.label_encoder(obesity_eating, ["target",
                                                             "Gender",
                                                             "family_history_with_overweight",
                                                             "FAVC",
                                                             "SMOKE",
                                                             "SCC",
                                                             "CAEC"])
        obesity_eating = self.ordinal_encoder(obesity_eating, ["FCVC",
                                                               "CALC"])
        obesity_eating = self.one_hot_encode(obesity_eating, ["MTRANS"])

        heart_failure = datasets['heart_failure_clinical_records_dataset'].copy()

        hepatitis = datasets['hepatitis'].copy()
        hepatitis = hepatitis.replace("?", np.nan).dropna()

        tictac_toe = datasets['tic-tac-toe'].copy()
        tictac_toe = self.one_hot_encode(tictac_toe, ["A1",
                                                      "A2",
                                                      "A3",
                                                      "A4",
                                                      "A5",
                                                      "A6",
                                                      "A7",
                                                      "A8",
                                                      "A9"])
        tictac_toe = self.label_encoder(tictac_toe, ["target"])

        balance_scale = datasets['balance-scale'].copy()
        balance_scale = self.label_encoder(balance_scale, ["target"])

        dermatology_dataset = datasets['dermatology'].copy()
        dermatology_dataset = dermatology_dataset.replace("?", np.nan).dropna()

        ecoli_dataset = datasets['ecoli'].copy()
        ecoli_dataset = self.label_encoder(ecoli_dataset, ["target"]).drop(columns="ID")

        early_diabetes = datasets['early_stageDiabetes'].copy()
        early_diabetes = self.label_encoder(early_diabetes, ["Gender",
                                                             "Polyuria",
                                                             "Polydipsia",
                                                             "sudden weight loss",
                                                             "weakness",
                                                             "Polyphagia",
                                                             "Genital thrush",
                                                             "visual blurring",
                                                             "Itching",
                                                             "Irritability",
                                                             "delayed healing",
                                                             "partial paresis",
                                                             "muscle stiffness",
                                                             "Alopecia",
                                                             "Obesity",
                                                             "target"]
                                                             )


        contraceptive_method = datasets['cmc'].copy()

        higher_education = datasets['higher_education'].copy()
        higher_education = higher_education.drop(columns=["ID","COURSEID"])

        MONK_problem1 = datasets['monks1'].copy()
        MONK_problem1 = MONK_problem1.astype('float64')

        MONK_problem2 = datasets['monk2'].copy()
        MONK_problem2 = MONK_problem2.astype('float64')

        MONK_problem3 = datasets['monk3'].copy()
        MONK_problem3 = MONK_problem3.astype('float64')

        land_mines = datasets['Mine_Dataset'].copy()

        echocardiogram_df = datasets['echocardiogram'].copy()
        echocardiogram_df = echocardiogram_df.replace("?", np.nan).dropna()
        echocardiogram_df = echocardiogram_df.drop(columns=["name","group"])

        rice_df = datasets['Rice_Cammeo_Osmancik'].copy()
        rice_df = self.label_encoder(rice_df, ["target"])

        mushroom = datasets['agaricus-lepiota'].copy()
        mushroom = mushroom.replace("?", np.nan).dropna()
        mushroom = self.label_encoder(mushroom, ["target"])
        mushroom = self.one_hot_encode(mushroom, ["a0",
                                                  "a1",
                                                  "a2",
                                                  "a3",
                                                  "a4",
                                                  "a5",
                                                  "a6",
                                                  "a7",
                                                  "a8",
                                                  "a9",
                                                  "a10",
                                                  "a11",
                                                  "a12",
                                                  "a13",
                                                  "a14",
                                                  "a15",
                                                  "a16",
                                                  "a17",
                                                  "a18",
                                                  "a19",
                                                  "a20",
                                                  "a21",]).dropna()
        
        nursery = datasets['nursery'].copy()
        nursery = self.label_encoder(nursery, ["target",
                                               "finance"
                                               ])
        nursery = self.ordinal_encoder(nursery, ["form",
                                                 "children",
                                                 "health"
                                                 ])
        nursery = self.one_hot_encode(nursery, ["parents",
                                                "social",
                                                "has_nurs",
                                                "housing"])
        darwin_df = datasets['darwin'].copy()
        darwin_df = darwin_df.drop(columns="ID")
        darwin_df = self.label_encoder(darwin_df, ["target"])

        auction_verification_df = datasets['auction_verification'].copy()
        auction_verification_df = self.label_encoder(auction_verification_df, ["target"]).drop(columns="verificationtime")

        caesarin_df = datasets['caesarian'].copy()
        caesarin_df = caesarin_df.astype("float64")

        cryotherapy_df = datasets['Cryotherapy'].copy()
        
        sirtuin6_df = datasets['SIRTUIN6'].copy()
        sirtuin6_df = self.label_encoder(sirtuin6_df, ["target"])
        
        sani_dataset = datasets['extention of Z-Alizadeh sani dataset'].copy()
        sani_dataset = self.label_encoder(sani_dataset, ["target",
                                                         "LAD",
                                                         "Sex",
                                                         "Obesity",
                                                         "CRF",
                                                         "CVA",
                                                         "Airway disease",
                                                         "Thyroid Disease",
                                                         "CHF",
                                                         "DLP",
                                                         "Weak Peripheral Pulse",
                                                         "Lung rales",
                                                         "Systolic Murmur",
                                                         "Diastolic Murmur",
                                                         "Dyspnea",
                                                         "Atypical",
                                                         "Nonanginal",
                                                         "Exertional CP",
                                                         "LowTH Ang",
                                                         "LVH",
                                                         "Poor R Progression",
                                                         "BBB",
                                                         ])
        sani_dataset = self.ordinal_encoder(sani_dataset, ["VHD"])

        myocardial_df =  datasets['MI'].copy()
        myocardial_df = myocardial_df.astype("float64")

        lymphography_df =  datasets['dataset_10_lymph'].copy()
        lymphography_df = self.label_encoder(lymphography_df, ["target",
                                                               "block_of_affere",
                                                               "bl_of_lymph_c",
                                                               "bl_of_lymph_s",
                                                               "by_pass",
                                                               "extravasates",
                                                               "regeneration_of",
                                                               "early_uptake_in",
                                                               "dislocation_of",
                                                               "exclusion_of_no"])
        lymphography_df = self.ordinal_encoder(lymphography_df, ["lymphatics"])
        lymphography_df = self.one_hot_encode(lymphography_df, ["changes_in_lym",
                                                                "defect_in_node",
                                                                "changes_in_node",
                                                                "changes_in_stru",
                                                                "special_forms"])

        polish_companies1 =  datasets['1year'].copy()
        polish_companies1 = polish_companies1.dropna()
        polish_companies1["target"] = polish_companies1["target"].astype("float64")

        polish_companies2 =  datasets['2year'].copy()
        polish_companies2 = polish_companies2.dropna()
        polish_companies2["target"] = polish_companies2["target"].astype("float64")

        polish_companies3 =  datasets['3year'].copy()
        polish_companies3["target"] = polish_companies3["target"].astype("float64")

        polish_companies4 =  datasets['4year'].copy()
        polish_companies4["target"] = polish_companies4["target"].astype("float64")

        polish_companies5 =  datasets['5year'].copy()
        polish_companies5["target"] = polish_companies5["target"].astype("float64")

        cervical_center =  datasets['risk_factors_cervical_cancer'].copy()
        cervical_center = cervical_center.replace("?", np.nan).dropna()

        cardiovascular =  datasets['cardiovascular-disease'].copy()
        cardiovascular = cardiovascular.drop(columns="id")

        audiology_df = datasets['dataset_7_audiology'].copy()
        audiology_df = self.ordinal_encoder(audiology_df, audiology_df.columns)

        post_operative_df = datasets['post-operative'].copy()
        post_operative_df = post_operative_df.replace("?", np.nan).dropna()
        post_operative_df = self.ordinal_encoder(post_operative_df, ["a1",
                                                                     "a2",
                                                                     "a3",
                                                                     "a4",
                                                                     "a5",
                                                                     "a6",
                                                                     "a7",
                                                                     ])
        post_operative_df = self.label_encoder(post_operative_df, ["target"])

        bone_marrow = datasets['bone-marrow'].copy()
        bone_marrow = self.ordinal_encoder(bone_marrow, ["Recipientgender",
                                                         "Stemcellsource",
                                                         "Donorage35",
                                                         "IIIV",
                                                         "Gendermatch",
                                                         "RecipientRh",
                                                         "ABOmatch",
                                                         "CMVstatus",
                                                         "DonorCMV",
                                                         "RecipientCMV",
                                                         "Riskgroup",
                                                         "Txpostrelapse",
                                                         "Diseasegroup",
                                                         "HLAmatch",
                                                         "HLAmismatch",
                                                         "Antigen",
                                                         "Alel",
                                                         "HLAgrI",
                                                         "Recipientage10",
                                                         "Recipientageint",
                                                         "Relapse",
                                                         "aGvHDIIIIV",
                                                         "extcGvHD",
                                                         "target"
                                                         ])
        bone_marrow = self.one_hot_encode(bone_marrow, ["DonorABO",
                                                        "RecipientABO",
                                                        "Disease",]).dropna()

        user_knowledge_df = datasets['Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN'].copy()
        user_knowledge_df = self.label_encoder(user_knowledge_df, ["target"])

        student_performance = datasets['CEE_DATA'].copy()
        student_performance = self.ordinal_encoder(student_performance,["target",
                                                                        "Gender",
                                                                        "time",
                                                                        "Class_ X_Percentage",
                                                                        "Class_XII_Percentage"])
        student_performance = self.one_hot_encode(student_performance, ["Caste",
                                                                        "coaching",
                                                                        "Class_ten_education",
                                                                        "twelve_education",
                                                                        "medium",
                                                                        "Father_occupation",
                                                                        "Mother_occupation"])

        teach_assistant = datasets['tae'].copy()

        breast_tissue = datasets['BreastTissue'].copy()
        breast_tissue = self.label_encoder(breast_tissue, ["target"])

        divorce_predictors = datasets['divorce'].copy()

        qsar_biodregad = datasets['biodeg'].copy()
        qsar_biodregad = self.label_encoder(qsar_biodregad, ["target"])
        qsar_biodregad["target"] = qsar_biodregad["target"].astype("float64")

        turkish_music = datasets['Acoustic Features'].copy()
        turkish_music = self.label_encoder(turkish_music, ["target"])
        turkish_music["target"] = turkish_music["target"].astype("float64")

        steel_plates = datasets['steel-plates-fault'].copy()
        steel_plates["target"] = steel_plates["target"].astype("float64")

        krvvskp_df = datasets['dataset_3_kr-vs-kp'].copy()
        krvvskp_df = self.label_encoder(krvvskp_df, krvvskp_df.columns)

        phoneme_df = datasets['phoneme'].copy()
        phoneme_df["target"] = phoneme_df["target"].astype("float64")
        
        eeg_eye = datasets['EEG Eye State'].copy()
        eeg_eye["target"] = eeg_eye["target"].astype("float64")

        bank_marketing = datasets['bank_marketing'].copy()
        bank_marketing = self.label_encoder(bank_marketing, ["V5",
                                                             "V7",
                                                             "V8",
                                                             "target"
                                                             ])
        bank_marketing = self.one_hot_encode(bank_marketing, ["V2",
                                                              "V3",
                                                              "V9",
                                                              "V16"])
        bank_marketing = self.ordinal_encoder(bank_marketing, ["V4",
                                                               "V11"])

        abalone_df = datasets['dataset_abalone'].copy()
        abalone_df = self.label_encoder(abalone_df, ["target",
                                                     "Sex"])

        analcatdata_df = datasets['analcatdata_dmft'].copy()
        analcatdata_df = self.ordinal_encoder(analcatdata_df, ["Gender",
                                                               "Ethnic"])
        analcatdata_df = self.label_encoder(analcatdata_df, ["target"])

        proba_football = datasets['prob_sfootball'].copy()
        proba_football = self.label_encoder(proba_football, ["target",
                                                             "Overtime"]).drop(columns=["Weekday"])
        proba_football = self.one_hot_encode(proba_football, ["Favorite_Name",
                                                              "Underdog_name"])

        car_df = datasets['car'].copy()
        car_df = self.ordinal_encoder(car_df, car_df.columns)

        yeast_Df = datasets['dataset_185_yeast'].copy()
        yeast_Df = self.label_encoder(yeast_Df, ["target"])

        kropt = datasets['kropt'].copy()
        kropt = self.label_encoder(kropt, ["target"])
        kropt = self.ordinal_encoder(kropt, ["a1",
                                             "a3",
                                             "a5"])

        higgs_df = datasets['higgs'].copy()
        
        return (
            acute_inflammations,
            acute_nephritis,
            autism_df,
            autism_adoles_df,
            autism_child_df,
            blood_df,
            messidor_df,
            thyroid_df,
            fertility_df,
            haberman_df,
            immunotherapy_df,
            maternal_health_risk_df,
            npha_doctor_visits_df,
            sa_heart,
            vertebral_df,
            breast_cancer_wisconsin_df,
            pima_diabetes_df,
            bc_coimbra_df,
            indian_liver_df,
            parkinsons_df,
            mammographic_masses_df,
            hcv_egyptian_df,
            thoracic_surgery_df,
            glioma,
            cirrhosis_df,
            iris_df,
            wine_df,
            heart_cleveland,
            german_credit_df,
            bc_yugoslavia_df,
            student_dropout,
            glass_identification ,
            credit_approval_df ,
            thyrroid_disease ,
            obesity_eating,
            heart_failure,
            hepatitis,
            tictac_toe,
            balance_scale,
            dermatology_dataset,
            ecoli_dataset,
            early_diabetes,
            contraceptive_method,
            higher_education,
            MONK_problem1,
            MONK_problem2,
            MONK_problem3,
            land_mines,
            echocardiogram_df,
            rice_df,
            mushroom,
            nursery,
            darwin_df,
            auction_verification_df,
            caesarin_df,
            cryotherapy_df,
            sirtuin6_df,
            sani_dataset,
            myocardial_df,
            lymphography_df,
            polish_companies1,
            polish_companies2,
            polish_companies3,
            polish_companies4,
            polish_companies5,
            cervical_center,
            cardiovascular,
            audiology_df,
            post_operative_df,
            bone_marrow,
            user_knowledge_df,
            student_performance,
            teach_assistant,
            breast_tissue,
            divorce_predictors,
            qsar_biodregad,
            turkish_music,
            steel_plates,
            krvvskp_df,
            phoneme_df,
            eeg_eye,
            bank_marketing,
            abalone_df,
            analcatdata_df,
            proba_football,
            car_df,
            yeast_Df,
            kropt,
            higgs_df
        )
