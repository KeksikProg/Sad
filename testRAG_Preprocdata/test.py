from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Загружаем Chroma с указанием persist_directory
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(collection_name="rag_data", 
    embedding_function=embeddings, 
    persist_directory="./chroma_db",
)

# 🔍 Получаем первые 5 документов (можно фильтровать по metadata)
docs = vectorstore.similarity_search("Профили с уклоном в backend?", k=5)

for i, doc in enumerate(docs):
    print(f"\n📄 Документ {i+1}:\n{doc.page_content}\n🔖 Метаданные: {doc.metadata}")
