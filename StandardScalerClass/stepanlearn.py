import numpy as np


class StepanStandartizator:

    def __init__(self, copy=True):
        self._standardizer_dict = {}
        self.copy = copy

    @staticmethod
    def __is_numpy(x):
        return isinstance(x, np.ndarray)

    def fit(self, x, y=None):

        # Пустой список, который будет заполняться парами выборочных средних и 
        # несмещенных выборочных дисперсий
        # Флаг массива Numpy
        is_np = self.__is_numpy(x)

        # Гарантия двумерного массива
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        # Число столбцов
        ncols = x.shape[1]

        # Ветвление, возвращает пару выборочное среднее - несмещенная дисперсия 
        # Для numpy.ndarray или pd.DataFrame
        if is_np:
            for col in range(ncols):
                self._standardizer_dict[col] = [np.mean(x[:, col]), np.nanvar(x[:, col], ddof=0)]
        else:
            for col in x.columns:
                self._standardizer_dict[col] = [x[col].mean(), x[col].var()]

        return self

    def transform(self, x):

        if self.copy:
            x = x.copy()

        is_np = self.__is_numpy(x)

        # Гарантия двумерного массива
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        nrows = x.shape[0]
        # Число столбцов
        ncols = x.shape[1]
        if is_np:
            for col in range(ncols):
                for row in range(nrows):
                    x[row, col] = (x[row, col] - self._standardizer_dict[col][0]) / np.sqrt(
                        self._standardizer_dict[col][1])
        else:
            for col in x.columns:
                for row in range(nrows):
                    x.iloc[row, col] = (x.iloc[row, col] - self._standardizer_dict[col][0]) / np.sqrt(
                        self._standardizer_dict[col][1])

        return x
