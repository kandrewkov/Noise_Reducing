import os
import time
import numpy as np
from builtins import str
from aiogram import Bot, Dispatcher, executor, types
from config import TOKEN
from aiogram.types import ContentType, File
from src.tools import *
from tensorflow import keras
from scipy.io.wavfile import write
from pathlib import Path
# from aiogram.types.input_file import InputFile


def get_project_root() -> Path:
    return Path(__file__).parent


rate = 16000

bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)

model_path = os.path.join(get_project_root(), 'src', 'model', 'NoiseSuppressionModel.h5')
temp_files_path = "temp_files"

raw_voice_name = "raw_voice.wav"
freq_changed_voice_name = "freq16_voice.wav"
clean_voice_name = "clean_voice.wav"

raw_voice_path = os.path.join(temp_files_path, raw_voice_name)
freq_changed_voice_path = os.path.join(temp_files_path, freq_changed_voice_name)
clean_voice_path = os.path.join(temp_files_path, clean_voice_name)

model = keras.models.load_model(filepath=model_path)


async def handle_file(file: File, file_name: str, path: str):
    Path(f"{path}").mkdir(parents=True, exist_ok=True)
    await bot.download_file(file_path=file.file_path, destination=f"{path}/{file_name}")


@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    await message.reply("Hello world")
    while True:
        await message.reply('Hello')
        time.sleep(10)


@dp.message_handler(content_types=[ContentType.VOICE])
async def voice_message_handler(message: types.Message):
    print('Приняли голосове сообщение')
    file_id = message.voice.file_id
    file = await bot.get_file(file_id)
    file_path_on_tg_server = file.file_path
    await bot.download_file(file_path_on_tg_server, raw_voice_path)

    wav_to_16_kHz(raw_voice_name, freq_changed_voice_name, temp_files_path)
    no_noise_voice = predict(model, os.path.join(get_project_root(), freq_changed_voice_path))
    audio_tensor = tf.squeeze(no_noise_voice, axis=[-1])
    audio_tensor = np.array(audio_tensor)
    audio_tensor = audio_tensor / audio_tensor.max() * 32767
    audio_tensor = np.int16(audio_tensor)
    write(clean_voice_path, rate, audio_tensor)

    # plt.plot(audio_tensor)
    # plt.show()

    await bot.send_audio(message.from_user.id, audio=open(clean_voice_path, 'rb'), performer="Твой шумокиллер",
                         title="Чистое аудио")
    await bot.send_voice(message.from_user.id, voice=open(clean_voice_path, 'rb'))

    print('Готово')


if __name__ == '__main__':
    print('it works')
    executor.start_polling(dispatcher=dp)
