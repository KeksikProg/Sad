from aiogram import Bot, Dispatcher, types, Router
from aiogram.types import Message
import aiohttp
import asyncio
from dotenv import load_dotenv
import os
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

@router.message()
async def handle_message(message: Message):
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/chat", json={"message": message.text}) as resp:
            data = await resp.json()
            print(message.text)
            await message.answer(data["response"])

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())