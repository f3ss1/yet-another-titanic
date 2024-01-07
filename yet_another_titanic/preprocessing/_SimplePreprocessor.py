class SimplePreprocessor:
    def __init__(self, columns_to_mix, mixed_column_name):
        self.columns_to_mix = columns_to_mix
        self.mixed_column_name = mixed_column_name

    def fit(self, data):
        return self

    def transform(self, data):
        data = self._process_siblings(data)
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def _process_siblings(self, data):
        data[self.mixed_column_name] = data[self.columns_to_mix].sum(axis=1)
        return data.drop(self.columns_to_mix, axis=1)
