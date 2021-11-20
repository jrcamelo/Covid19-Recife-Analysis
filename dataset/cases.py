import pandas as pd

from dataset.base import DatasetBase
from dataset.symptom_normalizer import SymptomNormalizer
from dataset.disease_normalizer import DiseaseNormalizer
from column_names import RAW_DISEASES, RAW_OTHER_DISEASES, AGE, AGE_GROUP, GENDER, HEALTH_PROFESSIONAL, RAW_OTHER_SYMPTOMS, SEVERITY, RAW_EVOLUTION, RAW_DEATH_DATE, RAW_SYMPTOMS

DATA_FOLDER = './dataset/data/'
TRAIN_TO_TEST_RATIO = 3 / 4

class DatasetCases(DatasetBase):
    RAW_COLUMN_NAMES = []
    MERGED_COLUMN_NAMES = [
        'data_notificacao',
        'sexo',
        AGE,
        'data_inicio_sintomas',
        'raca',
        'etnia',
        'sintomas',
        'outros_sintomas',
        'doencas_preexistentes',
        'evolucao',
        'data_obito',
        'profissional_saude',
        'categoria_profissional',
        'municipio_notificacao',
        'bairro',
        'ds',
        'em_tratamento_domiciliar',
        'classificacao_final',
        'severidade',
    ]
    BLANK_COLUMN_DATA = {}
    COLUMNS_TO_DELETE_WITHOUT_PROCESSING = [
        'data_inicio_sintomas',
        'raca',
        'etnia',
        'municipio_notificacao',
        'bairro',
        'ds',
        'em_tratamento_domiciliar',
        'classificacao_final',
        'categoria_profissional',
    ]
        
        
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.file = DATA_FOLDER + filename
        self.read_csv(self.file)
        self.csv_to_df()
        self.manage_columns()
        self.manage_rows()
        print("Processing data...")
        self.process_data()
        self.set_column_types()
        self.df = self.df.infer_objects()

    def manage_columns(self):
        self.create_blank_columns_if_not_exist(self.RAW_COLUMN_NAMES, self.BLANK_COLUMN_DATA)
        self.rename_columns(self.RAW_COLUMN_NAMES, self.MERGED_COLUMN_NAMES)
        self.merge_disease_columns_if_present()
        self.delete_columns(self.COLUMNS_TO_DELETE_WITHOUT_PROCESSING)
        
    def merge_disease_columns_if_present(self):
        if (RAW_OTHER_DISEASES in self.df.columns):
            self.df[RAW_DISEASES].fillna("NENHUMA", inplace=True)
            self.df[RAW_OTHER_DISEASES].fillna("NENHUMA", inplace=True)
            self.df[RAW_DISEASES] = self.df[RAW_DISEASES] + ',' + self.df[RAW_OTHER_DISEASES]
            self.df.drop(columns=[RAW_OTHER_DISEASES], inplace=True)
        
    def manage_rows(self):
        self.run_on_objects.append(self.normalize_strings)
        self.run_on_objects.append(self.replace_ignored_values_with_none)        
        self.iterate_columns()
        self.delete_empty_ages()
        
    def replace_ignored_values_with_none(self, column):
        self.replace_value_in_column(column, ["IGN", "IGNORADO"], None)
        
    def process_data(self):
        self.blank_gender_to_undefined()
        self.health_professional_to_boolean()
        self.evolution_to_severity()
        self.death_date_to_severity()
        self.months_to_0_age()
        self.ages_to_age_group()
        self.process_symptoms()
        self.process_diseases()
        
    def blank_gender_to_undefined(self):
        self.df.loc[self.df[GENDER].isna(), GENDER] = 'INDEFINIDO'

        
    def health_professional_to_boolean(self):
        self.df.loc[self.df[HEALTH_PROFESSIONAL] == 'SIM', HEALTH_PROFESSIONAL] = True
        self.df.loc[(self.df[HEALTH_PROFESSIONAL] != True), HEALTH_PROFESSIONAL] = False
        
    def evolution_to_severity(self):
        # TODO: Should this be kept? It confuses the model sometimes
        # self.df.loc[(self.df[RAW_EVOLUTION].notna()) & (self.df[RAW_EVOLUTION].str.startswith('INTERNADO')), SEVERITY] = 'INTERNADO'
        self.df.drop(columns=[RAW_EVOLUTION], inplace=True)

    def death_date_to_severity(self):
        self.df.loc[self.df[RAW_DEATH_DATE].notna(), SEVERITY] = 'OBITO'
        self.df.drop(columns=[RAW_DEATH_DATE], inplace=True)
        
    def delete_empty_ages(self):
        print("Deleting empty ages: " + str(self.df[AGE].isna().sum()))
        self.df = self.df.dropna(subset=[AGE])
                
    def months_to_0_age(self):
        self.df.loc[self.df[AGE].str.endswith('ES'), AGE] = 0
        
    def ages_to_age_group(self):
        self.df[AGE] = self.df[AGE].astype(int, errors='ignore')
        self.df[AGE_GROUP] = 0
        self.df.loc[self.df[AGE] < 10, AGE_GROUP] = 0
        self.df.loc[(self.df[AGE] >= 10) & (self.df[AGE] < 20), AGE_GROUP] = 10
        self.df.loc[(self.df[AGE] >= 20) & (self.df[AGE] < 30), AGE_GROUP] = 20
        self.df.loc[(self.df[AGE] >= 30) & (self.df[AGE] < 40), AGE_GROUP] = 30
        self.df.loc[(self.df[AGE] >= 40) & (self.df[AGE] < 50), AGE_GROUP] = 40
        self.df.loc[(self.df[AGE] >= 50) & (self.df[AGE] < 60), AGE_GROUP] = 50
        self.df.loc[(self.df[AGE] >= 60) & (self.df[AGE] < 70), AGE_GROUP] = 60
        self.df.loc[(self.df[AGE] >= 70) & (self.df[AGE] < 80), AGE_GROUP] = 70
        self.df.loc[self.df[AGE] >= 80, AGE_GROUP] = 80
        self.df.loc[self.df[AGE] == 0, AGE_GROUP] = "0-9"
        self.df.loc[self.df[AGE] == 10, AGE_GROUP] = "10-19"
        self.df.loc[self.df[AGE] == 20, AGE_GROUP] = "20-29"
        self.df.loc[self.df[AGE] == 30, AGE_GROUP] = "30-39"
        self.df.loc[self.df[AGE] == 40, AGE_GROUP] = "40-49"
        self.df.loc[self.df[AGE] == 50, AGE_GROUP] = "50-59"
        self.df.loc[self.df[AGE] == 60, AGE_GROUP] = "60-69"
        self.df.loc[self.df[AGE] == 70, AGE_GROUP] = "70-79"
        self.df.loc[self.df[AGE] == 80, AGE_GROUP] = "80+"        
        
    def set_column_types(self):
        pass

    def process_symptoms(self):
        symptoms = self.df[[RAW_SYMPTOMS, RAW_OTHER_SYMPTOMS]]
        normalizer = SymptomNormalizer(symptoms, self.filename)
        clean_symptoms = normalizer.get_normalized_columns()
        self.df.drop(columns=[RAW_SYMPTOMS, RAW_OTHER_SYMPTOMS], inplace=True)
        self.df = pd.concat([self.df, clean_symptoms], axis=1)
        # TODO: Should have assintomaticos?
        # self.df.loc[self.df[NO_SYMPTOMS] == True, SEVERITY] = 'ASSINTOMATICO'
        # self.df.drop(columns=[NO_SYMPTOMS], inplace=True)
    
    def process_diseases(self):
        diseases = self.df[[RAW_DISEASES]]
        diseases.to_csv('./dataset/misc/diseasesoriginal_' + self.filename + '.csv', index=False)
        normalizer = DiseaseNormalizer(diseases, self.filename)
        clean_diseases = normalizer.get_normalized_columns()
        self.df.drop(columns=[RAW_DISEASES], inplace=True)
        self.df = pd.concat([self.df, clean_diseases], axis=1)
        