#!/usr/bin/env python
# coding: utf-8


import pandas as pd 
import datetime as dt
import requests
import numpy as np
from io import StringIO


class TickersData():
    def __init__(self, tickers, start_date='today', end_date=None, report=True):
        """
        Класс для подключения к Московской бирже.
        tickers - массив(list, np.array и т.д.) или str - массив тикеров или единичный тикер.
        start_date - по умолчанию 'today'(сегодняшняя дата с вычетом 3 лет) - дата, от которой надо выгружать данные.
        end_date - по умолчанию None (выгружать все имеющиеся данные, начиная со start_date и заканчивая сегодняшним днем).
        stard_date и end_date принимают даты в виде YYYY-MM-DD.
        report - True или False - флаг для сообщения о состоянии загрузки данных.
        """
        if type(tickers) == str:
            tickers = [tickers]
        self.tickers = tickers
        if start_date =='today':
            today = dt.datetime.today()
            start_date = '-'.join([str(today.year - 3), str(today.month), str(today.day)])
        self.start_date = start_date
        self.end_date = end_date
        self.report = report
        self.prefix = 'http://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/'
        self.data = {}
        self.fill_in_data()
        
    def fill_in_data(self):
        """
        Функция для загрузки данных по каждому тикеру в self.data.
        """
        for ticker in self.tickers:
            self.start = 0
            self.data[ticker] = self.generate_dataframes(ticker)

    def create_url(self, ticker, url_type='.csv'):
        """
        Функция для генерации URL для подключения к бирже.
        ticker - str, название тикера.
        url_type - формат файла, который хотим получить. По умолчанию .csv.
        """
        postfix = ticker + url_type +'?from=' + self.start_date
        if self.end_date:
             postfix += '&till=' + self.end_date
        return self.prefix + postfix + '&start=' + str(self.start)
    
    def make_dataframe(self, url):
        """
        Функция генерирует датафрейм по URL, причем длина датафрейма ограничена 100 объектами из-за ограничений биржи.
        По сути является промежуточной функцией для generate_dataframes.
        url - str, непосредственно сам URL для запроса.
        """
        request = requests.get(url)
        string = StringIO(request.text[9:])
        dataframe = pd.read_csv(string, sep=';')
        return dataframe 
    
    def generate_dataframes(self, ticker):
        """
        Функция для генерации полного датафрейма для конкретного тикера, начиная со start_date заканчивая end_date, если задано.
        ticker - str, название тикера.
        """
        url = self.create_url(ticker)
        dataframe = self.make_dataframe(url)
        
        if dataframe.shape[0] == 0:
            print('"ОШИБКА": %s не имеет данных в базе Мосбиржи'%(ticker))
            return
        
        if dataframe.shape[0] < 100:
            if self.report:
                print(ticker, 'shape =', dataframe.shape, 'OK')
            return dataframe
        
        while True:
            self.start += 100
            temp_url = self.create_url(ticker)
            temp_dataframe = self.make_dataframe(temp_url)
            if temp_dataframe.shape[0] == 0:
                break
            dataframe = pd.concat((dataframe, temp_dataframe))
        if self.report:
            print(ticker, 'shape =', dataframe.shape, 'OK')
        return dataframe

