from yet_another_titanic.preprocessing._DataCleaner import DataCleaner
from yet_another_titanic.preprocessing._SimplePreprocessor import SimplePreprocessor


class Pipeline:
    def __init__(self, columns_to_drop, columns_to_mix, mixed_column_name):
        self.cleaner = DataCleaner(columns_to_drop)
        self.processor = SimplePreprocessor(columns_to_mix, mixed_column_name)

    def fit(self, data):
        self.cleaner.fit(data)
        self.processor.fit(data)
        return self

    def transform(self, data, clean: bool = True, drop_na: bool = True):
        if clean:
            data = self.cleaner.transform(data, drop_na=drop_na)
        transformed = self.processor.transform(data)
        return transformed

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
