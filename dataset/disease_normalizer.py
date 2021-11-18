import pandas as pd
from collections import Iterable

MAIN_DISEASES = {
    "doenca_cardiaca_ou_vascular": ["DOENCAS CARDIACAS OU VASCULARES", "DOENCAS CARDIACAS CRONICAS"],
    "doenca_diabetes": ["DIABETES"],
    "doenca_has": ["HAS", "HIPERTENSAO ARTERIAL", "HIPERTENSAO", "HIPERTENSO", "HIPERTENSA", "HIPERTENSAO ARTERIAL SISTEMICA", "PORTADORA DE HIPERTENSAO ARTERIAL SISTEMICA", "HIPERTENSSO", "PACIENTE HIPERTENSO", "HIPERTENCAO", "INCLUINDO HIPERTENSAO"],
    "doenca_respiratoria_cronica": ["DOENCAS RESPIRATORIAS CRONICAS DESCOMPENSADAS", "DOENCAS RESPIRATORIAS CRONICAS", "DOENCA PULMONAR CRONICA", "ASMA", "ASMATICA", "ASMATICO", "DPOC"],
    "doenca_obesidade": ["OBESIDADE", "SOBREPESO/OBESIDADE"],
    "doenca_renal": ["DOENCAS RENAIS CRONICAS", "DOENCA RENAL CRONICA", "RENAL CRONICA", "DOENCAS RENAIS CRONICAS EM ESTAGIO AVANCADO (GRAUS 3"],
    "doenca_imunosupressao": ["IMUNOSSUPRESSAO"],
    "doenca_tabagista": ["TABAGISTA", "EX TABAGISTA", "TABAGISMO", "EX-TABAGISTA", "FUMANTE", "EX-FUMANTE"],
    "doenca_etilismo": ["ETILISTA", "ETILISMO", "EX ETILISTA", "EX-ETILISTA", "ETILISTA CRONICO", "ETILISMO CRONICO", "ALCOOLISMO"],
    "doenca_hepatica": ["DOENCA HEPATICA CRONICA"],
    "doenca_neurologica": ["DOENCA NEUROLOGICA CRONICA", "DOENCA NEUROLOGICA", "ALZHEIMER", "PARKINSON"],    
}

NOT_DISEASES = [
    "NENHUMA"
]
    

class DiseaseNormalizer:
    COL_DISEASES = "doencas_preexistentes"
    
    COL_DISEASE_OTHER = "doenca_outros"
    
    def __init__(self, original, filename):
        self.original = original
        self.df = original.copy()
        self.filename = filename       
        self.make_disease_column_hash()
        self.split_symbols()
        self.save_all_unique_diseases_as_txt()
        self.add_disease_columns()
        self.fill_disease_columns()
        self.df[self.COL_DISEASES].to_csv('./dataset/misc/diseases_' + self.filename + '.csv', index=False)
        self.delete_original_column()
        self.df.to_csv("./dataset/misc/diseases_normalized_" + filename + ".csv", index=False)
    
    def get_normalized_columns(self):
        return self.df
    
    def make_disease_column_hash(self):
        self.disease_column_hash = {}
        for column, values in MAIN_DISEASES.items():
            for disease in values:
                self.disease_column_hash[disease] = column
    
    def get_disease_column(self, disease):
        return self.disease_column_hash.get(disease, self.COL_DISEASE_OTHER)
    
    def split_symbols(self):
        separator = ", |,| / | \+ |\+| E |;|\/"
        self.df[self.COL_DISEASES] = self.df[self.COL_DISEASES].fillna("")
        self.df[self.COL_DISEASES] = self.df[self.COL_DISEASES].str.split(separator)
        self.df[self.COL_DISEASES] = self.df[self.COL_DISEASES].apply(lambda x: [item.strip() for item in x])
        self.df[self.COL_DISEASES] = self.df[self.COL_DISEASES].apply(lambda x: [item for item in x if item != ""])
        self.df[self.COL_DISEASES] = self.df[self.COL_DISEASES].apply(lambda x: [item.replace(".", "") for item in x])
        self.df[self.COL_DISEASES] = self.df[self.COL_DISEASES].apply(lambda x: sorted(x))
        
    def add_disease_columns(self):
        for column in MAIN_DISEASES:
            self.df[column] = False
        self.df[self.COL_DISEASE_OTHER] = False
        
    def fill_disease_columns(self):
        for index, row in self.df.iterrows():
            for disease in row[self.COL_DISEASES]:
                self.set_disease_column_to_true(index, disease)

                
    def set_disease_column_to_true(self, index, disease):
        if (disease in NOT_DISEASES):
            return
        self.df.at[index, self.get_disease_column(disease)] = True

    def delete_original_column(self):
        self.df.drop(columns=[self.COL_DISEASES], inplace=True)
    
    def save_all_unique_diseases_as_txt(self):
        unique_diseases = {}
        for diseases in self.df[self.COL_DISEASES]:
            if isinstance(diseases, Iterable):
                for sympt in diseases:
                    if (unique_diseases.get(sympt) is None):
                        unique_diseases[sympt] = 1
                    else:
                        unique_diseases[sympt] += 1
        with open('./dataset/misc/disease_count_' + self.filename + '.txt', 'w') as f:
            for key, value in sorted(unique_diseases.items(), key=lambda item: item[1], reverse=True):
                    f.write(str(value) + " --- " + key + "\n")
