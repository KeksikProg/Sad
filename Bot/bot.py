from aiogram import Bot, Dispatcher, Router, types
from aiogram.types import Message
import aiohttp
import asyncio
from dotenv import load_dotenv
import os
from aiogram.filters import CommandStart

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_URL = os.getenv("API_URL")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

@router.message(CommandStart())
async def on_start(message: Message):
    await message.answer(
        "Привет! Я помогу сравнить направления/профили/УГС.\n"
        "Например: «Сравни 09.03.01 и 09.03.02». "
        "Важно: сравнивать можно только объекты одного уровня."
    )

# Храним статусы пользователей: True — бот занят, False — свободен
user_busy = {}
@router.message()
async def handle_message(message: Message):
    user_id = str(message.from_user.id)

    # Если бот занят этим пользователем — отвечаем сразу
    if user_busy.get(user_id, False):
        await message.answer("⏳ Дождитесь окончания прошлого ответа")
        return

    user_busy[user_id] = True  # Помечаем как занятый

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8000/chat",
                json={"message": message.text, "user_id": user_id}
            ) as resp:
                data = await resp.json()
                await message.answer(data["response"])
    except Exception as e:
        await message.answer(f"Ошибка: {e}")
    finally:
        user_busy[user_id] = False  # Снимаем блокировку

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
