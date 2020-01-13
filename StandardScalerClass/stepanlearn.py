import numpy as np
import numba as nb
from math import isnan


# Ускоренная функция расчета среднего
# Модифицируем baseline-функцию учетом возможных пропусков в данных
@nb.jit(nopython=True)
def nb_mean(arr):
    return np.nansum(arr) / len(arr)

# Ускоренная функция расчета стандартного несмещенного отклонения
# Модифицируем baseline-функцию учетом возможных пропусков в данных
@nb.jit(nopython=True)
def nb_std(arr):
    l = arr.shape[0]
    mean = nb_mean(arr)
    sumsq = 0.0
    for i in range(l):
        # Нашел только в модуле math такую проверку на NaN. В Numpy аналогичный метод работает иначе.
        if not isnan(arr[i]):
            sumsq += (arr[i] - mean) ** 2
    result = np.sqrt(sumsq / (l - 1))
    return result    


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
                self._standardizer_dict[col] = [nb_mean(x[:, col]), nb_std(x[:, col])]
        else:
             # записываем список столбцов
            num_columns = x.select_dtypes(exclude='object').columns.tolist()
            # по каждому столбцу датафрейма pandas
            for col in num_columns:
                self._standardizer_dict[col] = [x[col].mean(), x[col].std()]

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
                    x[:, col] = (x[:, col] - self._standardizer_dict[col][0]) / self._standardizer_dict[col][1]
        else:
            num_columns = x.select_dtypes(exclude='object').columns.tolist()
            for col in num_columns:
                      x[col] = (x[col] - self._standardizer_dict[col][0]) / self._standardizer_dict[col][1]

        return x
