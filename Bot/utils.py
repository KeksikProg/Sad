
from fastapi import HTTPException
import asyncio
from prompts import system_prompt
from langchain_core.messages import SystemMessage

def _cfg(user_id: str) -> dict:
    return {"configurable": {"thread_id": user_id}}

def _level(code: str) -> int:
    return len(code.split("."))  # 2 -> УГС, 2 parts -> направление, 3 parts -> профиль

def _validate_codes(codes: list[str], KNOWN_CODES):
    if not (2 <= len(codes) <= 5):
        raise HTTPException(400, "Нужно 2–5 кодов одного уровня.")
    lvl = _level(codes[0])
    if any(_level(c) != lvl for c in codes):
        raise HTTPException(400, "Смешаны уровни. Выберите только УГС, или только направления, или только профили.")
    missing = [c for c in codes if c not in KNOWN_CODES]
    if missing:
        raise HTTPException(404, f"Нет данных для: {', '.join(missing)}")

def _get_lock(user_id: str, _user_locks) -> asyncio.Lock:
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]

async def ensure_system_prompt(user_id: str, agent, _user_locks):
    """
    Атомарно проверяет, есть ли системный промпт.
    Если нет — добавляет его через update_state (без вызова LLM).
    """
    config = _cfg(user_id)
    lock = _get_lock(user_id, _user_locks)
    async with lock:
        # Получаем текущее состояние
        try:
            state = agent.get_state(config)
        except Exception:
            state = None

        # Проверяем, есть ли уже нужный системный промпт
        if state is not None:
            messages = state.values.get("messages", [])
            already_has_prompt = any(
                isinstance(m, SystemMessage)
                and isinstance(m.content, str)
                and "[PROMPT_V1]" in m.content
                for m in messages
            )
            if already_has_prompt:
                return

        # Если нет — добавляем системное сообщение через update_state
        agent.update_state(
            config,
            {"messages": [SystemMessage(content=system_prompt)]}
        )