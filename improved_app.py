import os
import time
import streamlit as st
from qa_loader import load_qa_and_create_vectorstore
from rag_chain import generate_response
from dotenv import load_dotenv

# 🔹 Load environment variables
load_dotenv()

# 🔹 Minimal CSS - sadece mesaj balonları için gerekli olan
def load_css():
    st.markdown("""
    <style>
        /* WhatsApp benzeri mesaj balonları */
        .message {
            display: flex;
            margin-bottom: 10px;
        }
        
        .message-user {
            justify-content: flex-end;
        }
        
        .message-assistant {
            justify-content: flex-start;
        }
        
        .message-content {
            padding: 10px 15px;
            border-radius: 18px;
            max-width: 70%;
            word-wrap: break-word;
        }
        
        .user-content {
            border: 1px solid #E5E7EB;
            border-bottom-right-radius: 5px;
        }
        
        .assistant-content {
            border: 1px solid #E5E7EB;
            border-bottom-left-radius: 5px;
        }
        
        .timestamp {
            font-size: 0.7rem;
            color: #64748B;
            margin-top: 2px;
            text-align: right;
        }
        
        .empty-chat {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 50vh;
            color: #64748B;
        }
        
        .empty-chat img {
            width: 100px;
            margin-bottom: 20px;
            opacity: 0.7;
        }
    </style>
    """, unsafe_allow_html=True)

# 🔹 Streamlit Page Configuration
st.set_page_config(
    page_title="University AI Assistant", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load minimal CSS
load_css()

# 🔹 Sidebar with information
with st.sidebar:
    st.image("logo.png", width=200)
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    This AI assistant helps answer questions about Vistula University.
    Ask anything about admissions, courses, campus life, and more!
    """)
    
    st.markdown("### 🔗 Quick Links")
    st.markdown("[University Website](https://www.vistula.edu.pl/en)")
    st.markdown("[Student Portal](https://www.vistula.edu.pl/en/students)")
    st.markdown("[Contact Us](https://www.vistula.edu.pl/en/contact)")

# 🔹 Main content area
st.title("🎓 University AI Assistant")
st.subheader("Your personal guide to university information. Ask me anything!")

# 🔹 Retrieve Data (Cached for Performance)
@st.cache_resource
def get_retriever():
    return load_qa_and_create_vectorstore()

retriever = get_retriever()

if isinstance(retriever, tuple):  
    retriever = retriever[0]

# 🔹 Start or Load Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query" not in st.session_state:
    st.session_state.query = ""

if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# 🔹 Display Chat History in WhatsApp style
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-chat">
        <img src="https://cdn-icons-png.flaticon.com/512/1041/1041916.png">
        <p>Start a conversation by asking a question below!</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for entry in st.session_state.chat_history:
        # User message
        st.markdown(f"""
        <div class="message message-user">
            <div class="message-content user-content">
                {entry['question']}
                <div class="timestamp">You</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant message
        st.markdown(f"""
        <div class="message message-assistant">
            <div class="message-content assistant-content">
                {entry['answer']}
                <div class="timestamp">Assistant</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# 🔹 Form kullanarak giriş alanını ve gönderme işlemini yönet
with st.form(key="message_form", clear_on_submit=True):
    user_input = st.text_input("Type your message...", key="user_input")
    submit_button = st.form_submit_button("Send")

# Handle category selection from sidebar
if st.session_state.query:
    user_input = st.session_state.query
    st.session_state.query = ""  # Clear after using
    submit_button = True
else:
    submit_button = submit_button

# 🔹 Process When User Submits a Question
if submit_button and user_input and not st.session_state.processing_done:
    with st.spinner("🤖"):
        response = generate_response(retriever, user_input)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "question": user_input,
            "answer": response
        })
        
        # Mark processing as done to prevent reprocessing
        st.session_state.processing_done = True
        
        # Force a rerun to update the UI
        st.rerun()

# Reset processing flag when input changes
if 'previous_input' not in st.session_state:
    st.session_state.previous_input = ""
    
if st.session_state.previous_input != user_input:
    st.session_state.processing_done = False
    st.session_state.previous_input = user_input

# 🔹 Footer
st.markdown("© 2025 University AI Assistant | Powered by AI")