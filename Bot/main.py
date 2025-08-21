from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn
from memory import CappingMemorySaver
import asyncio
from fastapi.responses import StreamingResponse
import io, pickle, numpy as np
from utils_viz import project_2d, plot_positions_2d, plot_tfidf_bars, plot_topics_text, plot_similarity_heatmap
from models import CodesIn, Input
from utils import _cfg, ensure_system_prompt, _validate_codes, _level
from fastapi import HTTPException
from compare_codes import compare_codes
from prompts import prompt_codes
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

#---LLM, AGENTS, TOOLS---
load_dotenv()
app = FastAPI()
#llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=LLM_TEMPERATURE)
tools = []
check = CappingMemorySaver()
agent = create_react_agent(model=llm, tools=tools, checkpointer=check)
_user_locks: dict[str, asyncio.Lock] = {}
#---DATA---
with open("ABS_FULL.pkl", "rb") as f:
    PROFILE_DATA = pickle.load(f)  # { code: {"embedding": np.ndarray, "tfidf": [(word,score)], "topics": [...] } }
KNOWN_CODES = set(PROFILE_DATA.keys())
#------

#---Visualization---
@app.post("/viz/position.png")
async def viz_position(data: CodesIn):
    codes = [c.strip() for c in data.codes]
    _validate_codes(codes, KNOWN_CODES)
    embs = np.vstack([np.array(PROFILE_DATA[c]["embedding"]) for c in codes])
    coords = project_2d(embs)
    png = plot_positions_2d(coords, codes)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")
@app.post("/viz/heatmap.png")
async def viz_heatmap(data: CodesIn):
    codes = [c.strip() for c in data.codes]
    _validate_codes(codes, KNOWN_CODES)
    embs = np.vstack([np.array(PROFILE_DATA[c]["embedding"]) for c in codes])
    png = plot_similarity_heatmap(codes, embs)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")
@app.get("/viz/tfidf/{code}.png")
async def viz_tfidf(code: str):
    code = code.strip()
    if code not in PROFILE_DATA:
        raise HTTPException(404, f"Нет данных для: {code}")
    png = plot_tfidf_bars(PROFILE_DATA[code]["tfidf"], code)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")
@app.get("/viz/topics/{code}.png")
async def viz_topics(code: str):
    code = code.strip()
    if code not in PROFILE_DATA:
        raise HTTPException(404, f"Нет данных для: {code}")
    png = plot_topics_text(PROFILE_DATA[code]["topics"], code)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")
#------

#---Just Chat---
@app.post("/chat")
async def chat(data: Input):
    await ensure_system_prompt(data.user_id, agent, _user_locks)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=data.message)]},
        config=_cfg(data.user_id)
    )
    return {"response": result["messages"][-1].content}
#------

#---SystemEndPoint---
@app.get("/codes")
async def get_codes():
    return {"codes": sorted(KNOWN_CODES)}
#------

#---VersusDirections---
@app.post("/compare")
async def compare(payload: CodesIn):
    codes = [c.strip() for c in payload.codes]
    _validate_codes(codes, KNOWN_CODES)

    # считаем метрики только по этим кодам
    result = compare_codes(codes)

    # формируем текст LLM строго на основе этих метрик (без распознавания кодов)
    template = ChatPromptTemplate.from_template(prompt_codes)
    prompt_msgs = template.format_messages(
        codes=", ".join(result["codes"]),
        similarities="\n".join([f"{k}: {v}" for k, v in result["similarities"].items()]),
        tfidf="\n".join([
            f"{k}: {', '.join([f'{w} ({round(float(s), 2)})' for w, s in v])}"
            for k, v in result["tfidf"].items()
        ]),
        topics="\n".join([
            f"{k}: " + "; ".join([", ".join(t['keywords']) for t in v])
            for k, v in result["topics"].items()
        ]),
        summaries="\n".join([f"{k}: {v}" for k, v in result.get("summaries", {}).items()])
    )
    text = (await llm.ainvoke(prompt_msgs)).content
    return {"text": text, "codes": result["codes"], "level": _level(result["codes"][0])}
#------

if __name__ == "__main__":
    uvicorn.run("main:app", port = 8000, host = "localhost", reload=True) # reload = True, чтобы сервер перезагружался при определенных изменениях
    # uvicorn main:app --reload