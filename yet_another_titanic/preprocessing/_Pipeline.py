from yet_another_titanic.preprocessing import DataCleaner, SimplePreprocessor


class Pipeline:
    def __init__(self, columns_to_drop):
        self.cleaner = DataCleaner(columns_to_drop)
        self.processor = SimplePreprocessor()

    def fit(self, data):
        self.cleaner.fit(data)
        self.processor.fit(data)
        return self

    def transform(self, data, clean: bool = True):
        if clean:
            data = self.cleaner.transform(data)
        transformed = self.processor.transform(data)
        return transformed

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
