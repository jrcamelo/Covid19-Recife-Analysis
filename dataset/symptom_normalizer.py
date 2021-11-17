import pandas as pd
from collections import Iterable

# Built using unique count for every symptom
MAIN_SYMPTOMS = {
    "sintoma_tosse": ["TOSSE"],
    "sintoma_dispneia": ["DISPNEIA", "FALTA DE AR"],
    "sintoma_febre": ["FEBRE"],
    "sintoma_baixa_saturacao_o2": ["SATURACAO O2 < 95"],
    "sintoma_desconforto_respiratorio": ["DESCONFORTO RESPIRATORIO"],
    "sintoma_aperto_toracico": ["APERTO TORACICO"],
    "sintoma_dor_garganta": ["DOR DE GARGANTA"],
    "sintoma_cefaleia": ["CEFALEIA", "DOR DE CABECA"],
    "sintoma_mialgia": ["MIALGIA", "DOR NO CORPO", "ALGIA"],
    "sintoma_diarreia": ["DIARREIA"],
    "sintoma_coriza": ["CORIZA"],
    "sintoma_congestao_nasal": ["CONGESTAO NASAL", "OBSTRUCAO NASAL"],
    "sintoma_fadiga": ["CANSACO/FADIGA", "FADIGA", "CANSACO", "FADIGA/CANSACO", "ADINAMIA", "MOLEZA", "MOLEZA NO CORPO", "CORPO MOLE"],
    "sintoma_anosmia_ou_hiposmia": ["ALTERACAO/PERDA DE OLFATO E/OU PALADAR", "DISTURBIOS GUSTATIVOS", "DISTURBIOS OLFATIVOS", "PERDA DO OLFATO", "PERDA DE OLFATO", "OLFATO", "PERDA OLFATO", "PERDA DO OLFATO/ MIALGIA", "PERDA DO PALADAR", "PERDA DE PALADAR", "DO PALADAR", "PALADAR", "PERDA D PALADAR", "SEM PALADAR", "ERDA DO PALADAR", ],
    "sintoma_vomito": ["VOMITO", "VOMITOS"],
    "sintoma_astenia": ["ASTENIA", "FRAQUEZA"],
    "sintoma_nausea": ["NAUSEA", "NAUSEAS", "ENJOO"],
    "sintoma_inaptencia": ["INAPTENCIA", "FALTA DE APETITE", "ANOREXIA", "HIPOREXIA"],
    "sintoma_dor_abdominal": ["DOR ABDOMINAL"],
}

# Those will be ignored
NO_SYMPTOMS = [
    "",
    "0",
    "ASSINTOMATICO",
    "ASSINTOMATICA",
    "ASSINTOMATICOS",
    "ASS",
    "SEM SINTOMAS",
    "SEM SINTOMA",
    "SEM SINTOMAS REFERIDOS",
    "SEM",
    "SS",
    "S",
    "NENHUM",
    "NAO",
    "NI",
    "NAO INFORMADO",
    "N INFORMADO",
    "NAO TEVE SINTOMAS",
    "NAO TEVE",
    "ASSINTOMATICO PARA CIRURGIA",
    "ASSINTOMATICO - CIRURGIA SEGURA",
    "NAO ASSINTOMATICO",
    "PACIENTE ASSINTOMATICO COM EXPOSICAO DE RISCO HA 20 DIAS OU MAIS",
]

NOT_SYMPTOMS = [
    "EXAME PRE OPERATORIO",
    "PRE OPERATORIO",
    "PRE-OP",
    "CONTACTANTE",
    "CONTATO",
    "CONTATO COM POSITIVO",
    "CONTATO COM CASO POSITIVO",
    "CONTATO DE CASO CONFIRMADO",
    "CONTATO COM CASO CONFIRMADO",
    "CONTATO COM PACIENTE POSITIVADO",
    "O LABORATORIO NAO DISPOE INFORMACAO SOBRE SINTOMAS",
    "O LABORATORIO NAO DISPOE DE INFORMACOES SOBRE OS SINTOMAS",
    "O LABORATORIO NAO DISPOE INFORMACAO SOBRE OS SINTOMAS",
    "OBS: TESTE REALIZADO NO REAL LAB (RHP)",
    "COLETA DRIVE-THRUR",
    "COLETA DRIVE-THUR",
    "APENAS COLETA",
    "EM ANALISE",
    "BUSCA ATIVA PRESIDIO PAMFA",
    "BUSCA ATIVA PFDB",
    "TOMOU A VACINA",
    "BUSCA ATIVA CPFR",
    "ELETIVA",
]

class SymptomNormalizer:
    def __init__(self, original, filename):
        self.original = original
        self.df = original.copy()
        self.filename = filename
        self.make_symptom_column_hash()        
        self.split_symbols()
        self.save_all_unique_symptoms_as_txt()
        self.add_symptom_columns()
        self.fill_symptom_columns()
        self.delete_original_columns()
        # self.df.to_csv("./dataset/misc/symptoms_normalized_" + filename + ".csv", index=False)
    
    def get_normalized_columns(self):
        return self.df
    
    def make_symptom_column_hash(self):
        self.symptom_column_hash = {}
        for column, values in MAIN_SYMPTOMS.items():
            for symptom in values:
                self.symptom_column_hash[symptom] = column
    
    # TODO: Confirm if the 13000 random symptoms should be ignored or added to "outros"
    def get_symptom_column(self, symptom):
        return self.symptom_column_hash.get(symptom, "sintoma_outros")
    
    def split_symbols(self):
        symbols = ", |,| / | \+ |\+| E |;"
        self.split_and_trim('sintomas', symbols)
        self.split_and_trim('outros_sintomas', symbols)
    
    def split_and_trim(self, column, separator):
        self.df[column] = self.df[column].fillna("")
        self.df[column] = self.df[column].str.split(separator)
        self.df[column] = self.df[column].apply(lambda x: [item.strip() for item in x])
        self.df[column] = self.df[column].apply(lambda x: [item for item in x if item != ""])
        self.df[column] = self.df[column].apply(lambda x: [item.replace(".", "") for item in x])
        self.df[column] = self.df[column].apply(lambda x: sorted(x))
        
    def add_symptom_columns(self):
        for column in MAIN_SYMPTOMS:
            self.df[column] = False
        self.df['sintoma_outros'] = False
        self.df['assintomatico'] = True
        
    def fill_symptom_columns(self):
        for index, row in self.df.iterrows():
            for symptom in row["sintomas"]:
                if symptom in NO_SYMPTOMS:
                    break
                self.set_symptom_column_to_true(index, symptom)
            for symptom in row["outros_sintomas"]:
                if symptom in NO_SYMPTOMS:
                    break
                self.set_symptom_column_to_true(index, symptom)

                
    def set_symptom_column_to_true(self, index, symptom):
        if (symptom in NOT_SYMPTOMS):
            return
        self.df.at[index, self.get_symptom_column(symptom)] = True
        self.df.at[index, "assintomatico"] = False

    def delete_original_columns(self):
        self.df.drop(columns=['sintomas', 'outros_sintomas'], inplace=True)
    
    def save_all_unique_symptoms_as_txt(self):
        unique_symptoms = {}
        for column in self.df.columns:
            for symptoms in self.df[column]:
                if isinstance(symptoms, Iterable):
                    for sympt in symptoms:
                        if (unique_symptoms.get(sympt) is None):
                            unique_symptoms[sympt] = 1
                        else:
                            unique_symptoms[sympt] += 1        
        with open('./dataset/misc/symptom_count_' + self.filename + '.txt', 'w') as f:
            for key, value in sorted(unique_symptoms.items(), key=lambda item: item[1], reverse=True):
                    f.write(str(value) + " --- " + key + "\n")