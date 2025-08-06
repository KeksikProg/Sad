from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name="rag_data",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

prompt = ChatPromptTemplate.from_template(
    """Ты ассистент, который может отвечать на вопросы по дисциплинам.
Используй прилагающиеся куски контекста для ответа на вопросы. Если ты не знаешь ответа на вопросы, просто скажи "Я не знаю.".
Вопрос: {quest}
Контекст: {context}
Ответ:"""
)

model_id = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    device=-1  # CPU
)

langchain_llm = HuggingFacePipeline(pipeline=hf_pipeline)
llm = ChatHuggingFace(llm=langchain_llm)

quest = "Расскажи мне про методологию ARIS"
retrieve_docs = vectorstore.similarity_search(quest, k=3)
docs_context = "\n".join([doc.page_content for doc in retrieve_docs])
messages = prompt.invoke({"quest": quest, "context": docs_context})

# --- Генерация ответа ---
answer = llm.invoke(messages)
print(answer.content)