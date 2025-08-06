from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import compare_directions_by_codes
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import uvicorn
load_dotenv()


app = FastAPI()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)
tools = [compare_directions_by_codes]
agent = create_react_agent(model=llm, tools=tools)

class Input(BaseModel):
    message: str

@app.post("/chat")
async def chat(data: Input):
    response = await agent.ainvoke({
        "messages": [HumanMessage(content=data.message)]
    }, config={"callbacks": None})
    return {"response": response['messages'][-1].content}
# uvicorn main:app --reload

if __name__ == "__main__":
    uvicorn.run("main:app", port = 8000, host = "localhost", reload=True) # reload = True, чтобы сервер перезагружался при определенных изменениях