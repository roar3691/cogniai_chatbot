import os
import asyncio
import uuid
from dotenv import load_dotenv
import streamlit as st
from textblob import TextBlob
import datetime
from pymongo import MongoClient, ASCENDING
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter, deque
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from googleapiclient.discovery import build
import PyPDF2
import bcrypt
import re

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB Setup
client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=False)
db = client["best_chatbot_db"]
chat_collection = db["chat_history"]
profile_collection = db["user_profiles"]
chat_collection.create_index([("user_id", ASCENDING), ("conversation_id", ASCENDING), ("timestamp", ASCENDING)])

# LangChain Setup with DeepSeek
prompt_template = PromptTemplate(
    input_variables=["query", "mood", "genre", "history", "google_results", "file_content", "preferences"],
    template="""
    You are the ultimate AI chatbot, excelling in reasoning, creativity, and user engagement. The user query is: "{query}". They’re feeling {mood}, and their preferred genre is {genre}. Here’s the conversation history: {history}. Google search results: {google_results}. Uploaded file content: {file_content}. User preferences: {preferences}.

    Instructions:
    - Craft a detailed, imaginative response with a beginning, middle, and end if storytelling is requested.
    - Use advanced reasoning to answer questions logically and insightfully.
    - Incorporate web search results and file content when relevant.
    - Reflect the user’s mood and preferences (tone, detail, language, format).
    - If the query is vague, ask clarifying questions politely.
    - Add unexpected twists or creative flair where appropriate (temperature=0.9).
    - Format as requested (paragraphs or bullets).
    Respond now!
    """
)
memory = ConversationBufferMemory(memory_key="history")
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="deepseek/deepseek-chat",
    temperature=0.9  # High creativity and reasoning
)
chat_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

# Session State Initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=50)
if "query_processing" not in st.session_state:
    st.session_state.query_processing = False
if "last_proactive_time" not in st.session_state:
    st.session_state.last_proactive_time = 0
if "notifications" not in st.session_state:
    st.session_state.notifications = []

# Helper Functions
def detect_mood(user_input: str) -> str:
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity
    if "happy" in user_input.lower() or sentiment > 0.3:
        return "happy"
    elif "stressed" in user_input.lower() or "anxious" in user_input.lower() or sentiment < -0.3:
        return "stressed"
    elif "nostalgic" in user_input.lower() or "memory" in user_input.lower():
        return "nostalgic"
    return "neutral"

def heuristic_multi_scale_attention(query):
    words = query.lower().split()
    length = len(words)
    short_scale = min(1.0, length / 5)
    specific_keywords = {"what", "how", "why", "who", "where", "when", "explain", "search", "file"}
    mid_scale = sum(1 for word in words if word in specific_keywords) / max(1, length)
    long_scale = 1.0 if "?" in query or len(re.findall(r"\w+", query)) > 3 else 0.5
    return min(max(0.3 * short_scale + 0.4 * mid_scale + 0.3 * long_scale, 0.1), 1.0)

async def perform_google_search(query):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = await asyncio.to_thread(service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=5).execute)
        return "\n".join([f"- {item['title']}: {item['snippet']}" for item in res.get("items", [])]) or "No results found."
    except Exception as e:
        return f"❌ Google Search Error: {e}"

def process_uploaded_file(file):
    if file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text if text else "No text extracted."
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    return "Unsupported file type (only PDFs supported)."

def create_profile(username, password):
    user_id = str(uuid.uuid4())
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    profile = {
        "user_id": user_id,
        "username": username,
        "password": hashed_password,
        "preferences": {"tone": "formal", "detail_level": "medium", "language": "en", "format": "paragraph"},
        "interests": {},
        "created_at": datetime.utcnow().timestamp(),
        "query_count": 0
    }
    profile_collection.insert_one(profile)
    return user_id

def authenticate_user(username, password):
    user = profile_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        return user["user_id"]
    return None

def get_user_preferences(user_id):
    user = profile_collection.find_one({"user_id": user_id})
    return user.get("preferences", {"tone": "formal", "detail_level": "medium", "language": "en", "format": "paragraph"}) if user else {"tone": "formal", "detail_level": "medium", "language": "en", "format": "paragraph"}

def update_user_preferences(user_id, preferences):
    profile_collection.update_one({"user_id": user_id}, {"$set": {"preferences": preferences}})

def store_chat(user_id, conversation_id, query, response):
    chat_entry = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "user_message": query,
        "ai_response": response,
        "timestamp": datetime.utcnow().timestamp()
    }
    chat_collection.insert_one(chat_entry)

def fetch_chat_history(user_id, conversation_id, limit=20):
    return list(chat_collection.find({"user_id": user_id, "conversation_id": conversation_id}, {"_id": 0, "user_message": 1, "ai_response": 1}).sort("timestamp", -1).limit(limit))

def detect_user_interests(user_id, conversation_id):
    history = fetch_chat_history(user_id, conversation_id)
    interests = Counter()
    stop_words = set(stopwords.words('english'))
    for chat in history:
        words = word_tokenize(chat["user_message"].lower())
        interests.update([word for word in words if word.isalnum() and word not in stop_words])
    return dict(interests.most_common(3))

async def generate_proactive_suggestion(user_id, conversation_id):
    interests = detect_user_interests(user_id, conversation_id)
    if interests and random.random() > 0.3:
        top_interest = max(interests, key=interests.get)
        return f"Noticed you’re into {top_interest}. Want to explore that further?"
    return f"How about we discuss {random.choice(['latest tech', 'a story', 'a puzzle'])}?"

# UI Functions
def profile_ui():
    st.title("Best Chatbot Ever")
    if not st.session_state.user_id:
        action = st.radio("Choose an action:", ["Login", "Sign Up"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if action == "Sign Up" and st.button("Sign Up"):
            if profile_collection.find_one({"username": username}):
                st.error("Username exists!")
            else:
                user_id = create_profile(username, password)
                st.session_state.user_id = user_id
                st.session_state.conversation_id = str(uuid.uuid4())
                st.success("Profile created!")
                st.rerun()
        elif action == "Login" and st.button("Login"):
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.conversation_id = str(uuid.uuid4())
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials!")
    else:
        st.subheader(f"Welcome, {st.session_state.user_id[:8]}!")
        with st.expander("Preferences"):
            prefs = get_user_preferences(st.session_state.user_id)
            tone = st.selectbox("Tone", ["formal", "casual"], index=["formal", "casual"].index(prefs["tone"]))
            detail = st.selectbox("Detail Level", ["low", "medium", "high"], index=["low", "medium", "high"].index(prefs["detail_level"]))
            lang = st.selectbox("Language", ["en", "es", "fr"], index=["en", "es", "fr"].index(prefs["language"]))
            format = st.selectbox("Format", ["paragraph", "bullet"], index=["paragraph", "bullet"].index(prefs["format"]))
            if st.button("Update Preferences"):
                new_prefs = {"tone": tone, "detail_level": detail, "language": lang, "format": format}
                update_user_preferences(st.session_state.user_id, new_prefs)
                st.success("Preferences updated!")
        if st.button("Logout"):
            st.session_state.user_id = None
            st.session_state.conversation_id = None
            st.session_state.chat_history.clear()
            st.rerun()

async def chatbot_ui():
    if not st.session_state.user_id:
        st.warning("Please log in or sign up.")
        return

    with st.sidebar:
        st.subheader("Notifications")
        for notif in st.session_state.notifications[-3:]:
            st.info(notif)
        if st.button("Clear Notifications"):
            st.session_state.notifications.clear()
            st.rerun()

    current_time = datetime.utcnow().timestamp()
    if current_time - st.session_state.last_proactive_time > 30 and not st.session_state.query_processing:
        suggestion = await generate_proactive_suggestion(st.session_state.user_id, st.session_state.conversation_id)
        st.session_state.notifications.append(suggestion)
        st.session_state.last_proactive_time = current_time
        st.rerun()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    file_content = process_uploaded_file(uploaded_file) if uploaded_file else None

    genre = st.sidebar.selectbox("Genre (for stories)", ["Fantasy", "Sci-Fi", "Romance", "None"], index=3)
    query = st.chat_input("Ask me anything...")

    if query and not st.session_state.query_processing:
        st.session_state.query_processing = True
        mood = detect_mood(query)
        google_results = await perform_google_search(query)
        preferences = get_user_preferences(st.session_state.user_id)
        focus_score = heuristic_multi_scale_attention(query)

        with st.spinner("Thinking..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(chat_chain.arun(
                query=query,
                mood=mood,
                genre=genre if genre != "None" else "general",
                google_results=google_results,
                file_content=file_content or "None",
                preferences=str(preferences)
            ))
            loop.close()

        store_chat(st.session_state.user_id, st.session_state.conversation_id, query, response)
        st.session_state.chat_history.append({"User": query, "AI": response})
        st.session_state.query_processing = False
        st.rerun()

    st.subheader("Chat History")
    for chat in reversed(fetch_chat_history(st.session_state.user_id, st.session_state.conversation_id)):
        with st.chat_message("user"):
            st.markdown(f"**You**: {chat['user_message']}")
        with st.chat_message("ai"):
            st.markdown(f"**AI**: {chat['ai_response']}")

    with st.expander("Analytics"):
        history = fetch_chat_history(st.session_state.user_id, st.session_state.conversation_id)
        st.write(f"Total Interactions: {len(history)}")
        interests = detect_user_interests(st.session_state.user_id, st.session_state.conversation_id)
        st.write("Top Interests:", ", ".join([f"{k} ({v})" for k, v in interests.items()]))

# Main Execution
profile_ui()
asyncio.run(chatbot_ui())
