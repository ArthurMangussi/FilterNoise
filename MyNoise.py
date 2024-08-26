from utilsMscMsc.MyPreprocessing import PreprocessingDatasets
import numpy as np 

class Noise:
    def __init__(self, datasets:dict):
        # self._logger = MeLogger()
        self._prep = PreprocessingDatasets()
        self.datasets = datasets
        self.wiscosin = self.pre_processing_wiscosin()
        self.pima = self.pre_processing_pima()
        self.indian_liver = self.pre_processing_indianLiver()
        self.parkinsons = self.pre_processing_parkinsons()
        self.mammo_masses = self.pre_processing_mammo()
        self.thoracic = self.pre_processing_thoracic()
        self.diabetic = self.pre_processing_diabetic()
        self.german = self.pre_processing_german_credit()
        self.adult = self.pre_processing_adult()
        self.thyroid = self.pre_processing_thyroid()
        self.blood = self.pre_processing_blood()
        self.law = self.pre_processing_law()
        
    def pre_processing_wiscosin(self):
        breast_cancer_wisconsin_df = self.datasets["wiscosin"].copy()
        breast_cancer_wisconsin_df = breast_cancer_wisconsin_df.drop(columns="ID")
        breast_cancer_wisconsin_df = self._prep.label_encoder(
            breast_cancer_wisconsin_df, ["target"]
        )
        return breast_cancer_wisconsin_df
    
    def pre_processing_indianLiver(self):
        indian_liver_df = self.datasets["indian_liver"].copy()
        indian_liver_df = indian_liver_df.dropna()
        indian_liver_df = self._prep.label_encoder(indian_liver_df, ["Gender"])
        return indian_liver_df
    
    def pre_processing_parkinsons(self):
        parkinsons_df = self.datasets["parkinsons"].copy().drop(columns="name")        
        return parkinsons_df
    
    def pre_processing_mammo(self):
        mammographic_masses_df = self.datasets["mammographic_masses"].copy()
        mammographic_masses_df = (
            self.datasets["mammographic_masses"]
            .replace("?", np.nan)
            .dropna()
            .drop(columns="BI-RADS assessment")
            .reset_index(drop=True)
        )
        mammographic_masses_df["Age"] = mammographic_masses_df["Age"].astype("int64")
        mammographic_masses_df["Density"] = mammographic_masses_df["Density"].astype(
            "int64"
        )
        mammographic_masses_df = self._prep.one_hot_encode(
            mammographic_masses_df, ["Shape", "Margin"]
        )
        return mammographic_masses_df
    
    def pre_processing_thoracic(self):
        thoracic_surgery_df = self.datasets["ThoraricSurgery"].copy()
        thoracic_surgery_df = self._prep.label_encoder(
            self._prep.one_hot_encode(thoracic_surgery_df, ["DGN"]),
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

        thoracic_surgery_df = self._prep.ordinal_encoder(thoracic_surgery_df, ["PRE6","PRE14"])
        return thoracic_surgery_df
    
    def pre_processing_bcCoimbra(self):
        return self.datasets["bc_coimbra"].copy()
    
    def pre_processing_diabetic(self):
        messidor_df = self.datasets["messidor_features"].copy()
        messidor_df["target"] = messidor_df["target"].astype("int64")
        return messidor_df 
    
    def pre_processing_adult(self):
        df = self.datasets["adult-clean"].copy()
        df = self._prep.ordinal_encoder(df, ["age",
                                    "education",
                                    "occupation",
                                    "gender"])

        df = self._prep.one_hot_encode(df, ["workclass",
                                    "marital-status",
                                    "relationship",
                                    "native-country",
                                    "race"])
        df = self._prep.label_encoder(df, ["target"])
        return df
    
    def pre_processing_thyroid(self):
        thyroid_df = self.datasets["Thyroid_Diff"].copy()
        thyroid_df = self._prep.label_encoder(thyroid_df, ["target"])

        thyroid_df = self._prep.ordinal_encoder(thyroid_df, ["Physical Examination",
                                                             "Gender", 
                                                            "Smoking", 
                                                            "Hx Smoking",
                                                            "Hx Radiothreapy",
                                                            "Focality",
                                                            "Thyroid Function",
                                                            "Pathology",
                                                            "Risk",
                                                            "T",
                                                            "N",
                                                            "M",
                                                            "Stage",
                                                            "Response"])

        thyroid_df = self._prep.one_hot_encode(thyroid_df, ["Adenopathy"])
        return thyroid_df
    
    def pre_processing_blood(self):
        blood_df = self.datasets["blood-transfusion-service-center"].copy()
        blood_df["target"] = blood_df["target"].astype("int64")
        return blood_df
    
    def pre_processing_german_credit(self):
        german_credit_df = self.datasets['german'].copy()

        map_gender = {"A91":"male",
                      "A92":"female",
                      "A93":"male",
                      "A94":"male",
                      "A95":"female"}
        
        german_credit_df["personal-status-and-sex"] = german_credit_df["personal-status-and-sex"].map(map_gender)

        german_credit_df = self._prep.ordinal_encoder(german_credit_df, ["age",
                                                                         "checking-account",
                                                                         "savings-account",
                                                                         "employment-since",
                                                                         "telephone",
                                                                         "foreign-worker",
                                                                         "personal-status-and-sex"])
        german_credit_df = self._prep.label_encoder(german_credit_df, ["target"])
        
        german_credit_df = self._prep.one_hot_encode(german_credit_df, ["credit-history",
                                                                        "purpose",
                                                                        "other-debtors",
                                                                        "property",
                                                                        "other-installment",
                                                                        "housing", 
                                                                        "job"])

        return german_credit_df
    
    def pre_processing_law(self):
        law_df = self.datasets["law_school_clean"].copy()
        law_df = self._prep.label_encoder(law_df, ["target"])
        law_df = self._prep.one_hot_encode(law_df, ["fam_inc", 
                                                    "race"])

        return law_df
    
    def pre_processing_pima(self):
        pima_diabetes_df = self.datasets["pima_diabetes"].copy()
        return pima_diabetes_df
    
    
    def cria_tabela(self):
        tabela_resultados = {}

        tabela_resultados["datasets"] = [self.wiscosin,
                                        self.pima, 
                                        self.indian_liver, 
                                        self.parkinsons,
                                        self.mammo_masses,                                         
                                        self.thoracic,                                         
                                        self.diabetic,
                                        #self.german,
                                        #self.adult,
                                        self.thyroid, 
                                        self.blood,
                                        self.law]
            
        tabela_resultados["nome_datasets"] = ["wiscosin",
                                            "pima",                                            
                                            "indian_liver",
                                            "parkinsons",
                                            "mammographic_masses",                                            
                                            "thoracic_surgery",                                            
                                            "diabetic_retionapaty",
                                            #"german_credit",
                                            #"adult",
                                            "thyroid_recurrence",
                                            "blood_transfusion",
                                            "law"]
            
        #tabela_resultados["missing_rate"] = [5,10,20]

        return tabela_resultados
    

    