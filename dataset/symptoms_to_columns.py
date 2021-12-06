import pandas as pd
from collections import Iterable
from column_names import *


# Built using unique count for every symptom
MAIN_SYMPTOMS = {
    SYMPTOM_COUGH: ["TOSSE", "HEMOPTISE", "TOSSE PRODUTIVA", "GRIPADA"],
    SYMPTOM_COLD: ["ESPIRROS", "ESPIRRO", "ESPIRRANDO", "ESPIROS", "ESPIRO", "GRIPE", "GRIPAL", "SINTOMAS GRIPAIS", "SINDROME GRIPAL", "SINTOMAS DE GRIPE"],
    SYMPTOM_AIR_INSUFFICIENCY: ["DISPNEIA", "FALTA DE AR", "TAQUIPNEIA", "TAQUIDISPNEIA", "INSUFICIENCIA RESPIRATORIA"],
    SYMPTOM_FEVER: ["FEBRE"],
    SYMPTOM_LOW_OXYGEN_SATURATION: ["SATURACAO O2 < 95", "DESSATURACAO"],
    SYMPTOM_BREATHING_CONDITION: ["DESCONFORTO RESPIRATORIO", "BATIMENTO ASA DE NARIZ", "TIRAGEM INTERCOSTAL", "DESCONFORTO TORACICO"],
    SYMPTOM_TORACIC_APERTURE: ["APERTO TORACICO", "APERTO NO PEITO", "DOR TORAXICA", "DOR DORACICA", "DOR TORACICA", "DOR NO PEITO"],
    SYMPTOM_THROAT_CONDITION: ["DOR DE GARGANTA", "ODINOFAGIA"],
    SYMPTOM_HEADACHE: ["CEFALEIA", "DOR DE CABECA", "DORES NA CABECA"],
    SYMPTOM_BODY_PAIN: ["MIALGIA", "DOR MUSCULAR", "DOR NO CORPO", "DORES NO CORPO", "ALGIA", "DOR LOMBAR", "DOR NA REGIAO LOMBAR", "DOR NAS COSTAS", "QUEDA DO ESTADO GERAL", "QUEDA DO ESTADO EM GERAL", "QUEDA DE ESTADO GERAL", "DORSALGIA", "LOMBALGIA", "DOR EM MMII", "DORES EM MMII", "DOR", "ARTRALGIA", "DORES NAS COSTAS", "NO CORPO", "DOR NAS ARTICULACOES", "DORES MUSCULARES", "DOR CORPO", ],
    SYMPTOM_DIARRHEA: ["DIARREIA"],
    SYMPTOM_RUNNY_NOSE: ["CORIZA", "RINORREIA", "SECRECAO"],
    SYMPTOM_NOSE_CONGESTION: ["CONGESTAO NASAL", "OBSTRUCAO NASAL"],
    SYMPTOM_WEAKNESS: ["ASTENIA", "FRAQUEZA", "CANSACO/FADIGA", "FADIGA", "CANSACO", "FADIGA/CANSACO", "ADINAMIA", "MOLEZA", "MOLEZA NO CORPO", "CORPO MOLE", "MAL ESTAR", "INDISPOSICAO"],
    SYMPTOM_ANOSMIA_OR_HYPOSMIA: ["ALTERACAO/PERDA DE OLFATO E/OU PALADAR", "DISTURBIOS GUSTATIVOS", "DISTURBIOS OLFATIVOS", "PERDA DO OLFATO", "PERDA DE OLFATO", "OLFATO", "PERDA OLFATO", "PERDA DO OLFATO/ MIALGIA", "PERDA DO PALADAR", "PERDA DE PALADAR", "DO PALADAR", "PALADAR", "PERDA D PALADAR", "SEM PALADAR", "ERDA DO PALADAR", "AGEUSIA", "AUGESIA", "AGUESIA", "DISGEUSIA", "PERDA DE OLFATO (CHEIRO) E/OU PALADAR (GOSTO)", "FALTA DE PALADAR", "SEM OFATO"],
    SYMPTOM_NAUSEA: ["NAUSEA", "NAUSEAS", "ENJOO", "VOMITO", "VOMITOS", "EMESE", "ANSIA DE VOMITO", "TONTURA"],
    SYMPTOM_LACK_OF_APPETITE: ["INAPTENCIA", "FALTA DE APETITE", "ANOREXIA", "HIPOREXIA", "DIMINUICAO DO APETITE"],
    SYMPTOM_ABDOMINAL_PAIN: ["DOR ABDOMINAL", "DOR EPIGASTRICA"],
    SYMPTOM_CONSCIOUSNESS_DEGRADATION: ["RNC", "SONOLENCIA", "DESORIENTACAO", "PROSTRACAO", "PROSTACAO", "REBAIXAMENTO DO NIVEL DE CONSCIENCIA", "SINCOPE", "DESORIENTADO", "CONFUSAO", "AFASIA", "CONFUSAO MENTAL", "REBAIXAMENTO DO NIVEL DE CONCIENCIA", "PERDA PONDERAL", "PERDA DA CONSCIENCIA", "REBAIXAMENTO NIVEL DE CONSCIENCIA"],
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
    "OUTROS: PACIENTE ASSINTOMATICO",
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
    "CONTATO COM POSITIVADO",
    "CONTATO POSITIVO",
    "BUSCA ATIVA PJALLB",
    "INTERNAMENTO HOSPITALAR",
    "CIRURGIA SEGURA",
    "CIRURGIA",
    "INTERNACAO HOSPITALAR",
]

class SymptomsToColumns:
    
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
    
    def get_symptom_column(self, symptom):
        return self.symptom_column_hash.get(symptom, SYMPTOM_OTHER)
    
    def split_symbols(self):
        symbols = ", |,| / | \+ |\+| E |;"
        self.split_and_trim(RAW_SYMPTOMS, symbols)
        self.split_and_trim(RAW_OTHER_SYMPTOMS, symbols)
    
    def split_and_trim(self, column, separator):
        self.df[column] = self.df[column].fillna("")
        self.df[column] = self.df[column].str.split(separator)
        self.df[column] = self.df[column].apply(lambda x: [item.strip() for item in x])
        self.df[column] = self.df[column].apply(lambda x: [item for item in x if item != ""])
        self.df[column] = self.df[column].apply(lambda x: [item.replace(".", "") for item in x])
        self.df[column] = self.df[column].apply(lambda x: sorted(x))
        
    def add_symptom_columns(self):
        # TODO: Should have assintomaticos?
        # self.df[NO_SYMPTOMS] = True
        for column in MAIN_SYMPTOMS:
            self.df[column] = False
        self.df[SYMPTOM_OTHER] = False
        
    def fill_symptom_columns(self):
        for index, row in self.df.iterrows():
            for symptom in row[RAW_SYMPTOMS]:
                if symptom in NO_SYMPTOMS:
                    break
                self.set_symptom_column_to_true(index, symptom)
            for symptom in row[RAW_OTHER_SYMPTOMS]:
                if symptom in NO_SYMPTOMS:
                    break
                self.set_symptom_column_to_true(index, symptom)

                
    def set_symptom_column_to_true(self, index, symptom):
        if (symptom in NOT_SYMPTOMS):
            return
        self.df.at[index, self.get_symptom_column(symptom)] = True
        # TODO: Should have assintomaticos?
        # self.df.at[index, NO_SYMPTOMS] = False

    def delete_original_columns(self):
        self.df.drop(columns=[RAW_SYMPTOMS, RAW_OTHER_SYMPTOMS], inplace=True)
    
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
