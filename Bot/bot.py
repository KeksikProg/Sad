from aiogram import Bot, Dispatcher, Router
from aiogram.types import Message, InputMediaPhoto, BufferedInputFile
from aiogram.filters import CommandStart
import aiohttp
import asyncio
import os
import re
from dotenv import load_dotenv
from utils import _level

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_URL = os.getenv("API_URL", "http://localhost:8000")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
router = Router()
dp.include_router(router)

# Регулярка для УГС/направлений/профилей: XX / XX.XX / XX.XX.XX
CODE_RE = re.compile(r"\b\d{2}(?:\.\d{2})?(?:\.\d{2})?\b", re.U)

# Глобальные состояния
user_busy: dict[str, bool] = {}
KNOWN_CODES: set[str] = set()

def _filter_codes_local(text: str) -> list[str]:
    """
    Достаём до 5 кодов из текста, фильтруем по KNOWN_CODES и одному уровню.
    Это только hint — сервер всё равно валидирует повторно.
    """
    if not text:
        return []
    raw = CODE_RE.findall(text)
    # убираем дубли с сохранением порядка
    seen, uniq = set(), []
    for c in raw:
        if c in KNOWN_CODES and c not in seen:
            seen.add(c); uniq.append(c)
            if len(uniq) >= 5:
                break
    if len(uniq) < 2:
        return []
    lvl = _level(uniq[0])
    if any(_level(c) != lvl for c in uniq):
        return []
    return uniq

@router.message(CommandStart())
async def on_start(message: Message):
    await message.answer(
        "Привет! Я помогу сравнить направления/профили/УГС.\n"
        "Например: «Сравни 09.03.01 и 09.03.02».\n"
        "Важно: сравнивать можно только объекты одного уровня (профили с профилями, направления с направлениями, УГС с УГС)."
    )

@router.message()
async def handle_message(message: Message):
    user_id = str(message.from_user.id)

    if user_busy.get(user_id, False):
        await message.answer("⏳ Дождитесь окончания прошлого ответа")
        return
    user_busy[user_id] = True

    try:
        # 1) Пытаемся вытащить валидные коды из текста (локально, по whitelist)
        codes_hint = _filter_codes_local(message.text)

        async with aiohttp.ClientSession() as session:
            if codes_hint:
                # 2А) Если нашли валидные коды — сравнение
                async with session.post(
                    f"{API_URL}/compare",
                    json={"codes": codes_hint},
                    timeout=120
                ) as resp:
                    if resp.status == 200:
                        payload = await resp.json()
                        text = payload.get("text", "Готово.")
                        final_codes = payload.get("codes", [])

                        # отправляем текст
                        await message.answer(text)

                        # отправляем визуализации ТОЛЬКО для final_codes
                        if len(final_codes) >= 2:
                            media = []

                            # 2D проекция
                            await message.chat.do("upload_photo")
                            r = await session.post(f"{API_URL}/viz/position.png", json={"codes": final_codes})
                            png_bytes = await r.read()
                            if r.status == 200:
                                media.append(
                                    InputMediaPhoto(
                                        media=BufferedInputFile(png_bytes, filename="position.png"),
                                        caption="2D‑проекция"
                                    )
                                )

                            # Тепловая карта
                            await message.chat.do("upload_photo")
                            r = await session.post(f"{API_URL}/viz/heatmap.png", json={"codes": final_codes})
                            png_bytes = await r.read()
                            if r.status == 200:
                                media.append(
                                    InputMediaPhoto(
                                        media=BufferedInputFile(png_bytes, filename="heatmap.png"),
                                        caption="Косинусная близость"
                                    )
                                )

                            # По каждому коду — TF‑IDF и темы
                            for code in final_codes:
                                await message.chat.do("upload_photo")
                                tf = await session.get(f"{API_URL}/viz/tfidf/{code}.png")
                                tf_bytes = await tf.read()
                                if tf.status == 200:
                                    media.append(
                                        InputMediaPhoto(
                                            media=BufferedInputFile(tf_bytes, f"tfidf_{code}.png"),
                                            caption=f"TF‑IDF — {code}"
                                        )
                                    )

                                tp = await session.get(f"{API_URL}/viz/topics/{code}.png")
                                tp_bytes = await tp.read()
                                if tp.status == 200:
                                    media.append(
                                        InputMediaPhoto(
                                            media=BufferedInputFile(tp_bytes, f"topics_{code}.png"),
                                            caption=f"BERTopic — {code}"
                                        )
                                    )

                            if media:
                                await message.answer_media_group(media[:10])
                    else:
                        # если /compare вернул ошибку — fallback в чат
                        await message.answer("Не удалось сравнить коды, попробую ответить в общем режиме…")
                        async with session.post(
                            f"{API_URL}/chat",
                            json={"message": message.text, "user_id": user_id},
                            timeout=120
                        ) as r2:
                            data = await r2.json()
                            await message.answer(data["response"])
            else:
                # 2Б) Кодов не нашли — обычный чат
                async with session.post(
                    f"{API_URL}/chat",
                    json={"message": message.text, "user_id": user_id},
                    timeout=120
                ) as resp:
                    data = await resp.json()
                    await message.answer(data["response"])

    except Exception as e:
        await message.answer(f"Ошибка: {e}")
    finally:
        user_busy[user_id] = False

async def _refresh_known_codes():
    """Фоновая задача: периодически подтягивать белый список кодов с бэкенда."""
    global KNOWN_CODES
    await asyncio.sleep(1.0)
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{API_URL}/codes", timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        codes = set(data.get("codes", []))
                        if codes:
                            KNOWN_CODES = codes
        except Exception:
            pass
        await asyncio.sleep(60000)  # каждые 5 минут

async def main():
    # первичная загрузка KNOWN_CODES
    asyncio.create_task(_refresh_known_codes())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
