class SimplePreprocessor:

    def fit(self, data):
        return self

    def transform(self, data):
        data = self._process_siblings(data)
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def _process_siblings(data):
        data['Siblings'] = data['SibSp'] + data['Parch']
        return data.drop(['SibSp', 'Parch'], axis=1)
