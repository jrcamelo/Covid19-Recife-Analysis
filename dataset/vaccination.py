from dataset.base import DatasetBase

VACCINATION_FILENAME = "vacinados.csv"

class DatasetVaccination(DatasetBase):
    def __init__(self):
        super().__init__(VACCINATION_FILENAME)