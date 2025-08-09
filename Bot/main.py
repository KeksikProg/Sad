from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from tools import compare_directions_by_codes
import uvicorn
from memory import CappingMemorySaver

load_dotenv()

app = FastAPI()

# Настройка LLM и инструментов
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)
tools = [compare_directions_by_codes]
check = CappingMemorySaver()
agent = create_react_agent(model=llm, tools=tools, checkpointer=check)


# Pydantic модель запроса
class Input(BaseModel):
    message: str
    user_id: str

# Маршрут FastAPI
@app.post("/chat")
async def chat(data: Input):
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=data.message)]},
        config={"configurable": {"thread_id": data.user_id}}
    )
    return {"response": result["messages"][-1].content}
# uvicorn main:app --reload

if __name__ == "__main__":
    uvicorn.run("main:app", port = 8000, host = "localhost", reload=True) # reload = True, чтобы сервер перезагружался при определенных изменениях