import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tools import parse_code


df = pd.read_csv('data/clear_data.csv')
df = df.dropna(subset=["text"]).copy()
df["text"] = df["text"].astype(str)

parsed_df = df["code"].apply(parse_code).apply(pd.Series)
df = pd.concat([df, parsed_df], axis=1)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index = True
)

all_splits = []

for _, row in df.iterrows():
    chunks = splitter.create_documents(
        texts=[row["text"]],
        metadatas=[{
            "level": "discipline",
            "type": "annotation",
            "group_code": row["group_code"],
            "direction_code": row["direction_code"],
            "profile_code": row["profile_code"],
        }]
    )
    all_splits.extend(chunks)

print(f"Total splits: {len(all_splits)}")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(collection_name="rag_data", 
    embedding_function=embeddings, 
    persist_directory="./chroma_db",
)

ids = vectorstore.add_documents(all_splits)
print(f"Total persisted documents: {len(ids)}")