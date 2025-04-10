import json
import os
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 📌 Gelişmiş embedding modeli (Daha iyi kelime eşleşmesi sağlar)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    encode_kwargs={"normalize_embeddings": True}
)

# 📌 Q&A Verisini Yükle ve Böl
def load_qa_and_create_vectorstore():
    with open("MyQ&A_cleaned.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # 📌 Veriyi uygun formatta dönüştür
    documents = [
        Document(page_content=f"Question: {item['QUESTION']}\nAnswer: {item['ANSWER']}")
        for item in qa_data
    ]

    # 📌 Metinleri belirli parçalara ayırarak vektör veritabanına uygun hale getir
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)  # ✅ split_docs artık tanımlandı!

    # 📌 ChromaDB'yi oluştur ve verileri sakla
    vectordb = Chroma.from_documents(split_docs, embedding_model, persist_directory="./vistula_chroma")

    return vectordb.as_retriever()
