import telebot
from predictor import Predictor
from tickers import TickersData
import matplotlib.pyplot as plt


my_api = '1411503812:AAFYM2Fld2VRnZmTS_TVcfvIBzBiDshHD9s'
bot = telebot.TeleBot(my_api)
print('GO')


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, введи тикер компании. Если не знаешь,'
                                      ' можно поискать тут : https://ffin.ru/market/directory/data/.')


@bot.message_handler(content_types=['text'])
def send_text(message):
    ticker = message.text.strip()
    ticker_data = TickersData(ticker)
    if ticker_data.data[ticker] is not None:
        try:
            bot.send_message(message.chat.id, f'Тикер {message.text}. Загружаю данные.🐊🐊🐊')
            predictor = Predictor([ticker], ticker_data)
            bot.send_message(message.chat.id, f'Цена сегодня: {predictor.today_price[ticker]:.3f} \n' +
                                              f'Цена на конец следующего квартала: {predictor.march_price[ticker]:.3f} \n' +
                                              f'Разница в цене: {predictor.sorted_prices[ticker]:.3f} \n' +
                                              f'MAE: {predictor.errors:.3f} руб.')
            plt.clf()
            plt.style.use('ggplot')
            plt.plot(predictor.tickers_predict[ticker], label='predict')
            plt.plot(predictor.stock_info[1:].iloc[:, 0].values, label='actual')
            plt.xticks(ticks=[8, 200, 380], labels=['2020-01', '2020-07', '2021-01'])
            plt.xlabel('Date')
            plt.ylabel('Price, RUB');
            plt.legend()
            plt.savefig(f'{message.chat.id}.png')
            with open(f'{message.chat.id}.png', 'rb') as img:
                bot.send_photo(message.chat.id, img)
        except UnboundLocalError:
            bot.send_message(message.chat.id, f'Данные по тикеру {message.text} за последний год не были найдены')

    else:
        bot.send_message(message.chat.id, f'По тикеру {ticker} не было найдено ничего на сайте Мосбиржи')



bot.polling()