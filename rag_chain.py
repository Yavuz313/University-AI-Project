import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
#from tavily import TavilyClient
from dotenv import load_dotenv
import datetime

# 🔹 Load environment variables from .env file
load_dotenv()

# 🔹 Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

#if not OPENAI_API_KEY or not TAVILY_API_KEY:
#    raise ValueError("❌ API keys are missing! Please check your .env file.")

# 🔹 Initialize OpenAI and Tavily clients
#tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

llm = ChatOpenAI(
    model_name="llama3-8b-8192",
    temperature=0,
    streaming=False,  # Streaming is controlled by Streamlit
    openai_api_key=OPENAI_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1"
)

# 🔎 Web search function using Tavily API
#def search_web_with_tavily(query):
#    if len(query) < 5:  # Ignore very short queries
#        return ""
#    
#    print(f"🔍 Sending query to Tavily: {query}")
#    search_results = tavily_client.search(query=query, max_results=3)
#
#    # Extract and format the retrieved web results
#    snippets = [f"{result['title']}: {result['content']}" for result in search_results['results'] if 'content' in result]
#    
#    print("✅ Web search results retrieved!")
#    return "\n".join(snippets) if snippets else "" 

# 📝 Prompt function for AI response generation
def prompt_fn(query: str, context: str, web_context: str = "") -> str:
    """
    This is the main prompt template for the AI assistant.
    
    The assistant must:
    - Prioritize university knowledge first.
    - Use web search only if internal knowledge is insufficient.
    - If no relevant information is found, respond with:
      "I’m sorry, but I don’t have information on this topic."
    - Avoid unnecessary introductions, greetings, or explanations.
    """

    # Include web search results only if available
    #search_part = f"\nAdditionally, I found the following information from the web:\n{web_context}\n" if web_context else ""

    return f"""
    Below is the available information for answering student inquiries about Vistula University.

    🔹 Follow this order when answering:
    1️⃣ **Use internal university knowledge first.**  
    2️⃣ **If internal data lacks relevant details.**  
    3️⃣ **If no useful information is found, respond with: "I’m sorry, but I don’t have information on this topic."**  

    🔹 Important Rules:
    - **Do not start with introductions.** Provide the answer directly.  
    - **If no information is available, do not add lengthy explanations.**  
    - **Never make up or guess information.**  

    🔹 Available Information:
    {context}
    
    🔹 Question:
    {query}

    ---
    ❗ **If no relevant information is found, simply say:**
    - "I’m sorry, but I don’t have information on this topic."
    """

# 🔹 Define the AI pipeline (Prompt → LLM → Output Parsing)
prompt_runnable = RunnableLambda(lambda inputs: prompt_fn(inputs["query"], inputs["context"], inputs.get("web_context", "")))
rag_chain = prompt_runnable | llm | StrOutputParser()

# 🔥 Response generation function
def generate_response(retriever, query):
    # 📌 Eğer soru çok kısa veya selamlaşma ise, özel bir yanıt ver
    greetings = ["hi", "hello", "hey", "merhaba", "greetings"]
    if query.lower().strip() in greetings:
        return "👋 Hello! How can I assist you today?"

    # 📌 AI veritabanında eşleşen yanıtları getir
    relevant_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # 📌 Eğer hiçbir veri bulunamazsa, AI modeline yönlendirme yap
    if not relevant_docs or len(context.strip()) < 20:
        return "I’m sorry, but I don’t have information on this topic."

    # 📌 AI Modeli ile yanıt oluştur
    inputs = {"query": query, "context": context}
    response = rag_chain.invoke(inputs).strip()

    # 📌 Eğer AI modeli boş veya gereksiz bir yanıt verirse, veritabanına dön
    if not response or "I don't know" in response.lower():
        return context if context else "I’m sorry, but I don’t have information on this topic."

    return response



# 🔹 Logging function for tracking interactions
def log_interaction(question, answer, source):
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)  # Ensure logs directory exists

    log_file = os.path.join(log_folder, "chat_log.txt")

    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Add timestamp
        f.write(f"{timestamp} | Question: {question}\n")  # Log user question
        f.write(f"{timestamp} | Answer: {answer}\n")  # Log AI response
        f.write(f"{timestamp} | Source: {source}\n")  # Indicate data source (VectorStore/Web)
        f.write("-" * 80 + "\n")  # Separator for readability
