from random import shuffle
import sqlite3
from openpyxl import load_workbook
from openpyxl.cell import Cell
from aiogram import Bot, Dispatcher, F
from aiogram.fsm.state import default_state, State, StatesGroup
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.types import Message, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
import os
from openpyxl.utils import column_index_from_string
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, create_engine
from model_SPAM_or_TRUE import spam_model
dp = Dispatcher()
bot = Bot(token='')
new_spam = ''
def spam_or_no():
    keyboard = InlineKeyboardBuilder()
    b1 = InlineKeyboardButton(text='Спам', callback_data='spam')
    b2 = InlineKeyboardButton(text='не спам', callback_data='not_spam')
    keyboard.add(b1, b2)
    return keyboard.adjust().as_markup()

@dp.message()
async def get_and_parse(message: Message):
    message_for_parse = message.text
    print(f'text={message_for_parse}')
    if len(message_for_parse) >= 10:
        prediction, probabilities = spam_model.predict(message_for_parse)
        print(prediction)
        if prediction == 1:
            uid = message.from_user.id
            uname = message.from_user.username
            await bot.send_message(text=f'Обнаружен спам от {uname}\n'
                                        f'Текст: {message_for_parse}', chat_id='1889078494', reply_markup=spam_or_no())
            global new_spam
            new_spam = message_for_parse



































if __name__ == '__main__':
    dp.run_polling(bot)