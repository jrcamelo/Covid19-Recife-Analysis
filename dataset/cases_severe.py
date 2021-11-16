from dataset.cases import DatasetCases

SEVERE_CASES_FILENAME = "casosgravescovid.csv"

class DatasetSevere(DatasetCases):
    RAW_COLUMN_NAMES = [
        'data_notificacao',
        'sexo',
        'idade',
        'data_inicio_sintomas',
        'raca',
        'etnia',
        'sintomas_apresentados',
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
        'severidade'
    ]
    BLANK_COLUMN_DATA = {'severidade': 'grave'}
    
    def __init__(self):
        super().__init__(SEVERE_CASES_FILENAME)