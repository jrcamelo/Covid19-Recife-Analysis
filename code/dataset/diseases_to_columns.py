import pandas as pd
from collections import Iterable
from column_names import *

MAIN_DISEASES = {
    DISEASE_HEART_OR_VASCULAR: ["DOENCAS CARDIACAS OU VASCULARES", "DOENCAS CARDIACAS CRONICAS", "CARDIOPATIA", "CARDIOPATA", "DISLIPIDEMIA", "ICC", "DOENCA CARDIOVASCULAR CRONICA", "DAC", "DOENCA CARDIOVASCULAR", "TVP", "TROMBOSE"],
    DISEASE_DIABETES: ["DIABETES", "DM"],
    DISEASE_HYPERTENSION: ["HAS", "HIPERTENSAO ARTERIAL", "HIPERTENSAO", "HIPERTENSO", "HIPERTENSA", "HIPERTENSAO ARTERIAL SISTEMICA", "PORTADORA DE HIPERTENSAO ARTERIAL SISTEMICA", "HIPERTENSSO", "PACIENTE HIPERTENSO", "HIPERTENCAO", "INCLUINDO HIPERTENSAO"],
    DISEASE_RESPIRATORY: ["DOENCAS RESPIRATORIAS CRONICAS DESCOMPENSADAS", "DOENCAS RESPIRATORIAS CRONICAS", "DOENCA PULMONAR CRONICA", "ASMA", "ASMATICA", "ASMATICO", "DPOC", "TUBERCULOSE", "PNEUMOPATIA CRONICA", "TUBERCULOSE PULMONAR", "OUTRA PNEUMOPATIA CRONICA", "PNEUMONIA CRONICA", "CANCER DE PULMAO"],
    DISEASE_OBESITY: ["OBESIDADE", "SOBREPESO/OBESIDADE"],
    DISEASE_KIDNEY: ["DOENCAS RENAIS CRONICAS", "DRC", "DOENCA RENAL CRONICA", "RENAL CRONICA", "DOENCAS RENAIS CRONICAS EM ESTAGIO AVANCADO (GRAUS 3", "4 OU 5)"],
    DISEASE_IMMUNOSUPPRESSION: ["IMUNOSSUPRESSAO", "IMUNODEFICIENCIA", "IMUNODEPRESSAO"],
    DISEASE_TABAGIST: ["TABAGISTA", "EX TABAGISTA", "TABAGISMO", "EX-TABAGISTA", "FUMANTE", "EX-FUMANTE", "EX-TABAGISMO", "EX- TABAGISTA", "EX FUMANTE", "HISTORICO DE TABAGISMO", "EX TABAGISMO"],
    DISEASE_ETILISM: ["ETILISTA", "ETILISMO", "EX ETILISTA", "EX-ETILISTA", "ETILISTA CRONICO", "ETILISMO CRONICO", "ALCOOLISMO", "ETILISTA CRONICO"],
    DISEASE_LIVER: ["DOENCA HEPATICA CRONICA"],
    DISEASE_NEUROLOGIC: ["DOENCA NEUROLOGICA CRONICA", "DOENCA NEUROLOGICA", "ALZHEIMER", "PARKINSON", "DEPRESSAO", "EPILEPSIA", "DEMENCIA", "ESQUIZOFRENIA", "SINDROME DEMENCIAL", "TRANSTORNO MENTAL", "DOENCAS NEUROLOGICAS", "SINDROME DE DOWN", "DISTURBIO PSIQUIATRICO", "ANSIEDADE", "MAL DE ALZHEIMER", "TRANSTORNO PSIQUIATRICO"],
}


NOT_DISEASES = [
    "NENHUMA"
]
    

class DiseasesToColumns:
    
    def __init__(self, original, filename):
        self.original = original
        self.df = original.copy()
        self.filename = filename       
        self.make_disease_column_hash()
        self.split_symbols()
        self.save_all_unique_diseases_as_txt()
        self.add_disease_columns()
        self.fill_disease_columns()
        self.df[RAW_DISEASES].to_csv('./dataset/misc/diseases_' + self.filename + '.csv', index=False)
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
        return self.disease_column_hash.get(disease, DISEASE_OTHER)
    
    def split_symbols(self):
        separator = ", |,| / | \+ |\+| E |;|\/"
        self.df[RAW_DISEASES] = self.df[RAW_DISEASES].fillna("")
        self.df[RAW_DISEASES] = self.df[RAW_DISEASES].str.split(separator)
        self.df[RAW_DISEASES] = self.df[RAW_DISEASES].apply(lambda x: [item.strip() for item in x])
        self.df[RAW_DISEASES] = self.df[RAW_DISEASES].apply(lambda x: [item for item in x if item != ""])
        self.df[RAW_DISEASES] = self.df[RAW_DISEASES].apply(lambda x: [item.replace(".", "") for item in x])
        self.df[RAW_DISEASES] = self.df[RAW_DISEASES].apply(lambda x: sorted(x))
        
    def add_disease_columns(self):
        for column in MAIN_DISEASES:
            self.df[column] = False
        self.df[DISEASE_OTHER] = False
        
    def fill_disease_columns(self):
        for index, row in self.df.iterrows():
            for disease in row[RAW_DISEASES]:
                self.set_disease_column_to_true(index, disease)

                
    def set_disease_column_to_true(self, index, disease):
        if (disease in NOT_DISEASES):
            return
        self.df.at[index, self.get_disease_column(disease)] = True

    def delete_original_column(self):
        self.df.drop(columns=[RAW_DISEASES], inplace=True)
    
    def save_all_unique_diseases_as_txt(self):
        unique_diseases = {}
        for diseases in self.df[RAW_DISEASES]:
            if isinstance(diseases, Iterable):
                for sympt in diseases:
                    if (unique_diseases.get(sympt) is None):
                        unique_diseases[sympt] = 1
                    else:
                        unique_diseases[sympt] += 1
        with open('./dataset/misc/disease_count_' + self.filename + '.txt', 'w') as f:
            for key, value in sorted(unique_diseases.items(), key=lambda item: item[1], reverse=True):
                    f.write(str(value) + " --- " + key + "\n")
