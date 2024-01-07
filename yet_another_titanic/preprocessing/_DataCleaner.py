class DataCleaner:
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, data):
        return self

    def transform(self, data):
        data = data.drop(self.columns_to_drop, axis=1, errors="ignore")
        data = data.dropna()
        return data
