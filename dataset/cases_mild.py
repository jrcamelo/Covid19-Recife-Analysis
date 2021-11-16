from dataset.cases import DatasetCases

MILD_CASES_FILENAME = "casoslevescovid.csv"

class DatasetMild(DatasetCases):
    RAW_COLUMN_NAMES = [
        'data_notificacao',
        'sexo',
        'idade',
        'data_inicio_sintomas',
        'raca_cor',
        'etnia',
        'sintomas',
        'outros_sintomas',
        'doencas_preexistentes',
        'evolucao_caso',
        'data_obito',
        'profissional_saude',
        'cbo',
        'municipio_notificacao',
        'bairro',
        'ds',
        'em_tratamento_domiciliar',
        'classificacao_final',
        'severidade'
    ]
    BLANK_COLUMN_DATA = {'severidade': 'leve'}
    
    def __init__(self):
        super().__init__(MILD_CASES_FILENAME)
        