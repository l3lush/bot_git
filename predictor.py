import pandas as pd
import datetime as dt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

LAG_START = 2
LAG_END = 15


class Predictor:
    def __init__(self, tickers, data):
        self.create_dates()
        self.tickers = tickers
        self.create_stock_info(data)
        self.data = pd.DataFrame(self.stock_info.iloc[1:, 0])
        self.prepareData()
        self.generate_data_for_next_quarter()
        self.ticker_predict()
        self.count_price_change()

    def create_dates(self):
        today = dt.datetime.today()
        self.year_ago = '-'.join((str(today.year - 1), str(today.month), str(today.day)))

        sdate = dt.date(today.year - 1, today.month, today.day)
        edate = dt.date(today.year, today.month, today.day)

        delta = edate - sdate
        days_for_past_year = []
        for i in range(delta.days + 1):
            days_for_past_year.append(sdate + dt.timedelta(days=i))

        self.dates = pd.DataFrame(days_for_past_year, index=days_for_past_year, columns=['TRADEDATE'])

    def create_stock_info(self, a):
        begin = True
        final_tickers = []
        for tick in self.tickers:
            if a.data[tick] is not None:
                data = a.data[tick][a.data[tick]['TRADEDATE'] > self.year_ago][['TRADEDATE', 'LEGALCLOSEPRICE']]
                if data.empty:
                    print('%s не имеет записей за последний год' % (tick))
                    continue
                if sum(data['LEGALCLOSEPRICE'].isna()) == len(data['LEGALCLOSEPRICE']):
                    print('%s полностью состоит из NaN' % (tick))
                    continue
                final_price = data.iloc[-1]['LEGALCLOSEPRICE']
                data.loc[:, 'TRADEDATE'] = pd.to_datetime(data.loc[:, 'TRADEDATE'])
                dates_n_price = self.dates.set_index('TRADEDATE').join(data.set_index('TRADEDATE'))
                dates_n_price.fillna(method='ffill', inplace=True)
                dates_n_price.fillna(method='bfill', inplace=True)
                date_window = dates_n_price.rolling('14d').mean()
                date_window.columns = ['_'.join((tick, '14-days-window'))]
                dates_n_price.columns = ['_'.join((tick, 'day-to-day-price'))]
                final_df = dates_n_price.join(date_window)
                dict_for_final_price = {final_df.columns[0]: [final_price], final_df.columns[1]: [final_price]}
                final_price_df = pd.DataFrame(dict_for_final_price, index=['final_price'])
                final_df = pd.concat((final_price_df, final_df))
                if begin:
                    stock_info = final_df
                    begin = False
                else:
                    stock_info = stock_info.join(final_df)
                final_tickers.append(tick)
        self.stock_info = stock_info

    def prepareData(self, lag_start=5, lag_end=20, test_size=0.15):
        """
        Функция для преобразования данных (добавление фич).
        data - данные.
        lag_start - номер первого лага, добавленного в признаки.
        lag_end - номер последнего лага, добавленного в признаки.
        test_size - доля тестовой выборки.
        """

        data = pd.DataFrame(self.data.copy())
        data.columns = ["y"]

        # считаем индекс в датафрейме, после которого начинается тестовыый отрезок
        test_index = int(len(data) * (1 - test_size))

        # добавляем лаги исходного ряда в качестве признаков
        for i in range(lag_start, lag_end):
            data["lag_{}".format(i)] = data.y.shift(i)
        data_index = pd.to_datetime(data.index)

        data["weekday"] = data_index.weekday
        data['is_weekend'] = data_index.weekday.isin([5, 6]) * 1
        data['year'] = data_index.year
        data['month'] = data_index.month
        data['day'] = data_index.day
        data['week'] = data_index.isocalendar().week
        data['day of year'] = data_index.dayofyear
        data['quarter'] = data_index.quarter

        data = data.dropna()
        data = data.reset_index(drop=True)

        # разбиваем весь датасет на тренировочную и тестовую выборку
        self.X_train = data.loc[:test_index].drop(["y"], axis=1)
        self.y_train = data.loc[:test_index]["y"]
        self.X_test = data.loc[test_index:].drop(["y"], axis=1)
        self.y_test = data.loc[test_index:]["y"]

    def generate_data_for_next_quarter(self):
        today = dt.datetime.today()
        # генерация массива дат, начиная с завтрашнего дня, заканчивая последним днем следующего квартала.
        tomorrow = dt.date(today.year, today.month, today.day + 1)
        tomorrow_quarter = pd.to_datetime(tomorrow).quarter
        if tomorrow_quarter != 4:
            next_quarter = tomorrow_quarter + 1
        else:
            next_quarter = 1
        if next_quarter == 4:
            future_quarter = 1
        else:
            future_quarter = next_quarter + 1
        sdate = pd.to_datetime(tomorrow)
        days_to_prognose = []
        while pd.to_datetime(sdate).quarter != future_quarter:
            days_to_prognose.append(sdate)
            sdate += dt.timedelta(days=1)
            sdate = pd.to_datetime(sdate)
        self.days_to_prognose = days_to_prognose

    def ticker_predict(self):
        tickers_predict = {}
        for i in range(1, self.stock_info.shape[1], 2):
            ticker = self.stock_info.iloc[1:, i].name.split('_')[0]
            X_train, y_train = cvartal_preparation(self.stock_info.iloc[1:, i])
            ranfor = RandomForestRegressor(n_estimators=30)
            self.errors = performTimeSeriesCV(self.X_train, self.y_train, 5, ranfor, mean_absolute_error)
            ranfor.fit(X_train, y_train)
            y = y_train.values
            for date in self.days_to_prognose:
                X = make_row(date, y)
                y_pred = ranfor.predict(X)
                y = np.append(y, y_pred)
            tickers_predict[ticker] = y.copy()
        self.tickers_predict = tickers_predict

    def count_price_change(self):
        # считаю предсказанное изменение цены
        price_change = {}
        self.today_price = dict()
        self.march_price = dict()
        for key, value in self.tickers_predict.items():
            march = value[-1]
            today = value[-len(self.days_to_prognose)]
            self.today_price[key] = today
            self.march_price[key] = march
            price_change[key] = march - today
        self.sorted_prices = {k: v for k, v in sorted(price_change.items(), key=lambda item: item[1], reverse=True)}


def performTimeSeriesCV(X_train, y_train, number_folds, model, metrics):
    """
    Кросс-валидация для временного ряда.
    X_train, y_train - данные и ответы для тренировки.
    number_folds - int, количество фолдов.
    model - сама модель.
    metrics - метрика качества.
    """

    k = int(np.floor(float(X_train.shape[0]) / number_folds))
    errors = np.zeros(number_folds - 1)
    for i in range(2, number_folds + 1):
        split = float(i - 1) / i

        X = X_train[:(k * i)]
        y = y_train[:(k * i)]

        index = int(np.floor(X.shape[0] * split))

        X_trainFolds = X[:index]
        y_trainFolds = y[:index]

        X_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]

        model.fit(X_trainFolds, y_trainFolds)
        errors[i - 2] = metrics(model.predict(X_testFold), y_testFold)

    return errors.mean()


def cvartal_preparation(data, lag_start=LAG_START, lag_end=LAG_END):
    """
    Функция для подготовки данных для предсказания на следующий квартал.
    data - данные.
    lag_start - номер первого лага, добавленного в признаки.
    lag_end - номер последнего лага, добавленного в признаки.
    """

    data = pd.DataFrame(data.copy())
    data.columns = ["y"]

    # добавляем лаги исходного ряда в качестве признаков
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.y.shift(i)

    data_index = pd.to_datetime(data.index)

    data["weekday"] = data_index.weekday
    data['is_weekend'] = data_index.weekday.isin([5, 6]) * 1
    data['year'] = data_index.year
    data['month'] = data_index.month
    data['day'] = data_index.day
    data['week'] = data_index.isocalendar().week
    data['day of year'] = data_index.dayofyear
    data['quarter'] = data_index.quarter

    data = data.dropna()
    data = data.reset_index(drop=True)

    # разбиваем весь датасет на тренировочную и тестовую выборку
    X_train = data.loc[:].drop(["y"], axis=1)
    y_train = data.loc[:]["y"]

    return X_train, y_train


def make_row(date, y):
    """
    Функция для создания ряда для данных из будущего.
    date - дата.
    y - массив ответов за предыдущие даты.
    """
    X = np.array([])
    for i in range(LAG_START, LAG_END):
        X = np.append(X, y[-i])
    X = np.append(X, date.weekday())
    X = np.append(X, 1 if date.weekday() in [5, 6] else 0)
    X = np.append(X, date.year)
    X = np.append(X, date.month)
    X = np.append(X, date.day)
    X = np.append(X, date.weekofyear)
    X = np.append(X, date.dayofyear)
    X = np.append(X, date.quarter)
    return X.reshape(1, -1)

