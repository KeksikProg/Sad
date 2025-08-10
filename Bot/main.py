from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from tools import compare_directions_by_codes
import uvicorn
from memory import CappingMemorySaver
from prompts import system_prompt
import asyncio


def _cfg(user_id: str) -> dict:
    return {"configurable": {"thread_id": user_id}}


load_dotenv()

app = FastAPI()

# Настройка LLM и инструментов
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)
tools = [compare_directions_by_codes]
check = CappingMemorySaver()
agent = create_react_agent(model=llm, tools=tools, checkpointer=check)


_user_locks: dict[str, asyncio.Lock] = {}
def _get_lock(user_id: str) -> asyncio.Lock:
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]

async def ensure_system_prompt(user_id: str):
    """
    Атомарно проверяет, есть ли системный промпт.
    Если нет — добавляет его через update_state (без вызова LLM).
    """
    config = _cfg(user_id)
    lock = _get_lock(user_id)
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

# Pydantic модель запроса
class Input(BaseModel):
    message: str
    user_id: str

# Маршрут FastAPI
@app.post("/chat")
async def chat(data: Input):
    await ensure_system_prompt(data.user_id)
    #print(agent.get_state({"configurable": {"thread_id": data.user_id}}))
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=data.message)]},
        config=_cfg(data.user_id)
    )
    return {"response": result["messages"][-1].content}
# uvicorn main:app --reload

if __name__ == "__main__":
    uvicorn.run("main:app", port = 8000, host = "localhost", reload=True) # reload = True, чтобы сервер перезагружался при определенных изменениях