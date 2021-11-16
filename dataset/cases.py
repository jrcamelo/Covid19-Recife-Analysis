import pandas as pd
from dataset.base import DatasetBase

DATA_FOLDER = './dataset/data/'
TRAIN_TO_TEST_RATIO = 3 / 4

class DatasetCases(DatasetBase):
    RAW_COLUMN_NAMES = []
    MERGED_COLUMN_NAMES = [
        'data_notificacao',
        'sexo',
        'idade',
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
        'categoria_profissional'
    ]
        
        
    def __init__(self, filename):
        super().__init__()
        self.file = DATA_FOLDER + filename
        self.read_csv(self.file)
        self.csv_to_df()
        
        self.create_blank_columns_if_not_exist(self.RAW_COLUMN_NAMES, self.BLANK_COLUMN_DATA)
        self.rename_columns(self.RAW_COLUMN_NAMES, self.MERGED_COLUMN_NAMES)
        self.delete_columns(self.COLUMNS_TO_DELETE_WITHOUT_PROCESSING)
        
        self.run_on_objects.append(self.normalize_strings)
        self.run_on_objects.append(self.replace_ignored_values_with_none)        
        self.iterate_columns()
        
        self.delete_empty_ages()
        
        self.health_professional_to_boolean()
        self.evolution_to_severity()
        self.death_date_to_severity()
        self.months_to_0_age()
        
        self.set_column_types()
        self.df = self.df.infer_objects()

    def replace_ignored_values_with_none(self, column):
        self.replace_value_in_column(column, ["IGN", "IGNORADO"], None)
        
    def health_professional_to_boolean(self):
        self.df.loc[self.df['profissional_saude'] == 'SIM', 'profissional_saude'] = True
        self.df.loc[self.df['profissional_saude'] != 'SIM', 'profissional_saude'] = False
        
    def evolution_to_severity(self):
        self.df.loc[(self.df['evolucao'].notna()) & (self.df['evolucao'].str.startswith('INTERNADO')), 'severidade'] = 'INTERNADO'
        self.df.drop(columns=['evolucao'], inplace=True)

    def death_date_to_severity(self):
        self.df.loc[self.df['data_obito'].notna(), 'severidade'] = 'OBITO'
        self.df.drop(columns=['data_obito'], inplace=True)
        
    def delete_empty_ages(self):
        self.df = self.df.dropna(subset=['idade'])
                
    def months_to_0_age(self):
        self.df.loc[self.df['idade'].str.endswith('ES'), 'idade'] = 0
        
    def set_column_types(self):
        self.df['idade'] = self.df['idade'].astype(int, errors='ignore')
        