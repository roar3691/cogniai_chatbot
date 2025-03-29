import streamlit as st
from openai import OpenAI
import asyncio
from collections import deque
from pymongo import MongoClient, ASCENDING
from datetime import datetime, timedelta
from googleapiclient.discovery import build
import uuid
import hashlib
import random
import re
import PyPDF2
import base64
import requests

# Environment Variables (Set in Streamlit Cloud secrets)
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = st.secrets.get("SEARCH_ENGINE_ID")
MONGO_URI = st.secrets.get("MONGO_URI")
SITE_URL = st.secrets.get("SITE_URL")
SITE_NAME = st.secrets.get("SITE_NAME")

# OpenRouter Clients with Error Handling
try:
    gemini_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    deepseek_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
except TypeError as e:
    st.error(f"Failed to initialize OpenAI clients: {e}")
    st.stop()

# MongoDB Setup
client = MongoClient(MONGO_URI)
db = client["advanced_chatbot_db"]
chat_collection = db["chat_history"]
profile_collection = db["user_profiles"]
MAX_CHAT_HISTORY = 500
chat_collection.create_index([("user_id", ASCENDING), ("timestamp", ASCENDING)])

# Session State Initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=50)
if "query_processing" not in st.session_state:
    st.session_state.query_processing = False
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {"tone": "casual", "detail_level": "medium", "language": "en", "format": "paragraph"}
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "last_proactive_time" not in st.session_state:
    st.session_state.last_proactive_time = 0
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0

# UI Title
st.title("CogniChat Advanced")

# Heuristic Multi-Scale Attention
def heuristic_multi_scale_attention(query):
    words = query.lower().split()
    length = len(words)
    short_scale = min(1.0, length / 5)
    specific_keywords = {"what", "how", "why", "who", "where", "when", "explain", "describe", "search", "image", "think"}
    mid_scale = sum(1 for word in words if word in specific_keywords) / max(1, length)
    long_scale = 1.0 if "?" in query or len(re.findall(r"\w+", query)) > 3 else 0.5
    focus_score = (0.3 * short_scale + 0.4 * mid_scale + 0.3 * long_scale)
    return min(max(focus_score, 0.1), 1.0)

def adjust_focus_score(query, focus_score):
    words = query.lower().split()
    if any(word in words for word in ["what", "how", "why", "who", "where", "when"]):
        act_bonus = 0.2
        emotion_bonus = 0.1 if any(word in words for word in ["great", "good", "happy"]) else 0.0
    elif any(word in words for word in ["tell", "give", "show"]):
        act_bonus = 0.1
        emotion_bonus = 0.0
    else:
        act_bonus = -0.1
        emotion_bonus = 0.0 if "please" in words else -0.1
    return min(max(focus_score + act_bonus + emotion_bonus, 0.1), 1.0)

# Profile Management
def create_profile(username, password):
    user_id = str(uuid.uuid4())
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    profile = {
        "user_id": user_id,
        "username": username,
        "password": hashed_password,
        "preferences": {"tone": "casual", "detail_level": "medium", "language": "en", "format": "paragraph"},
        "interests": {},
        "query_count": 0,
        "last_query_time": 0
    }
    profile_collection.insert_one(profile)
    return user_id

def authenticate_user(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    user = profile_collection.find_one({"username": username, "password": hashed_password})
    return user["user_id"] if user else None

def get_user_preferences(user_id):
    user = profile_collection.find_one({"user_id": user_id})
    return user.get("preferences", {"tone": "casual", "detail_level": "medium", "language": "en", "format": "paragraph"}) if user else {"tone": "casual", "detail_level": "medium", "language": "en", "format": "paragraph"}

def update_user_preferences(user_id, preferences):
    profile_collection.update_one({"user_id": user_id}, {"$set": {"preferences": preferences}})

def update_query_count(user_id):
    current_time = datetime.utcnow().timestamp()
    profile_collection.update_one({"user_id": user_id}, {"$inc": {"query_count": 1}, "$set": {"last_query_time": current_time}})
    st.session_state.last_query_time = current_time

# Google Search Function
async def perform_google_search(query):
    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        res = await asyncio.to_thread(service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=3).execute)
        return "\n".join([f"- [{item['title']}]({item['link']})\n{item['snippet']}" for item in res.get("items", [])]) or "No results found."
    except Exception as e:
        return f"❌ Google Search Error: {e}"

# Chat History Management
def fetch_chat_history(user_id, limit=10):
    return list(chat_collection.find({"user_id": user_id}, {"_id": 0, "user": 1, "ai": 1, "timestamp": 1}).sort("timestamp", -1).limit(limit))

def store_chat(user_id, query, response):
    chat_entry = {"user_id": user_id, "user": query, "ai": response, "timestamp": datetime.utcnow().timestamp()}
    chat_collection.insert_one(chat_entry)
    if chat_collection.count_documents({"user_id": user_id}) > MAX_CHAT_HISTORY:
        oldest = chat_collection.find_one({"user_id": user_id}, sort=[("timestamp", ASCENDING)])
        chat_collection.delete_one({"_id": oldest["_id"]})

def summarize_history(user_id):
    history = fetch_chat_history(user_id, 20)
    if not history:
        return "No recent conversation to summarize."
    return "\n".join([f"Q: {chat['user'][:50]}...\nA: {chat['ai'][:50]}..." for chat in history])

# Interest Detection and Proactive Suggestions
def detect_user_interests(user_id):
    history = fetch_chat_history(user_id, 20)
    interests = {}
    for chat in history:
        words = chat["user"].lower().split()
        for word in words:
            if len(word) > 3:
                interests[word] = interests.get(word, 0) + 1
    return dict(sorted(interests.items(), key=lambda x: x[1], reverse=True)[:3])

async def proactive_suggestion(user_id):
    interests = detect_user_interests(user_id)
    if interests and random.random() > 0.3:
        top_interest = max(interests, key=interests.get)
        return f"Hey, noticed you’re into {top_interest}. Want to chat about it?"
    topics = ["latest tech", "random facts", "puzzles"]
    return f"How about we explore {random.choice(topics)}?"

# File and Image Processing
def process_file(file):
    if file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            return f"PDF Content: {text}" if text else "No text extracted from PDF."
        except Exception as e:
            return f"Error processing PDF: {e}"
    return "Unsupported file type (only PDFs supported)."

def process_image(file):
    return base64.b64encode(file.read()).decode("utf-8")

# AI Query Functions
async def query_gemini(query, user_id, image_data=None, file_content=None):
    past_context = summarize_history(user_id)
    preferences = get_user_preferences(user_id)
    focus_score = heuristic_multi_scale_attention(query)
    focus_score = adjust_focus_score(query, focus_score)
    
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "text", "text": f"""
    **Query**: {query}
    **Context**: {past_context}
    **Preferences**: Tone: {preferences['tone']}, Detail: {preferences['detail_level']}, Language: {preferences['language']}, Format: {preferences['format']}
    **Focus Score**: {focus_score:.2f}
    Respond accordingly, incorporating file content or image data if provided.
    """})

    if file_content:
        messages[0]["content"].append({"type": "text", "text": file_content})
    if image_data:
        messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

    response = gemini_client.chat.completions.create(
        extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_NAME},
        model="google/gemini-2.5-pro-exp-03-25:free",
        messages=messages
    )
    return response.choices[0].message.content.strip()

async def query_deepseek(query, user_id):
    past_context = summarize_history(user_id)
    preferences = get_user_preferences(user_id)
    focus_score = heuristic_multi_scale_attention(query)
    focus_score = adjust_focus_score(query, focus_score)
    
    response = deepseek_client.chat.completions.create(
        extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_NAME},
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[{"role": "user", "content": f"""
        **Query**: {query}
        **Context**: {past_context}
        **Preferences**: Tone: {preferences['tone']}, Detail: {preferences['detail_level']}
        **Focus Score**: {focus_score:.2f}
        Provide logical reasoning or deep analysis.
        """}]
    )
    return response.choices[0].message.content.strip()

async def generate_image(query):
    # Placeholder: Replace with actual image generation API if available
    return "Image generation not yet implemented. Describe what you'd like, and I’ll simulate a response!"

# Profile UI
def profile_ui():
    if not st.session_state.user_id:
        st.subheader("User Profile")
        action = st.radio("Choose an action:", ["Login", "Sign Up"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if action == "Sign Up" and st.button("Sign Up"):
            if profile_collection.find_one({"username": username}):
                st.error("Username exists!")
            else:
                user_id = create_profile(username, password)
                st.session_state.user_id = user_id
                st.session_state.user_preferences = get_user_preferences(user_id)
                st.success("Profile created!")
                st.rerun()
        elif action == "Login" and st.button("Login"):
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.user_preferences = get_user_preferences(user_id)
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials!")
    else:
        st.subheader(f"Welcome, User {st.session_state.user_id[:8]}!")
        with st.expander("Preferences"):
            tone = st.selectbox("Tone", ["formal", "casual"], index=["formal", "casual"].index(st.session_state.user_preferences["tone"]))
            detail = st.selectbox("Detail Level", ["low", "medium", "high"], index=["low", "medium", "high"].index(st.session_state.user_preferences["detail_level"]))
            lang = st.selectbox("Language", ["en", "es", "fr", "hi"], index=["en", "es", "fr", "hi"].index(st.session_state.user_preferences["language"]))
            format = st.selectbox("Format", ["paragraph", "bullet"], index=["paragraph", "bullet"].index(st.session_state.user_preferences["format"]))
            if st.button("Update Preferences"):
                new_prefs = {"tone": tone, "detail_level": detail, "language": lang, "format": format}
                update_user_preferences(st.session_state.user_id, new_prefs)
                st.session_state.user_preferences = new_prefs
                st.success("Preferences updated!")
        if st.button("Logout"):
            st.session_state.user_id = None
            st.session_state.chat_history.clear()
            st.rerun()

# Chatbot UI
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

    user = profile_collection.find_one({"user_id": st.session_state.user_id})
    current_time = datetime.utcnow().timestamp()
    if user and (current_time - user.get("last_query_time", 0) < 2 or user.get("query_count", 0) > 50):
        st.warning("Please wait or limit reached.")
        return

    if current_time - st.session_state.last_proactive_time > 30 and not st.session_state.query_processing:
        suggestion = await proactive_suggestion(st.session_state.user_id)
        st.session_state.notifications.append(f"AI Suggests: {suggestion}")
        st.session_state.last_proactive_time = current_time
        st.rerun()

    uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "jpg", "jpeg", "png"])
    file_content = process_file(uploaded_file) if uploaded_file and uploaded_file.type == "application/pdf" else None
    image_data = process_image(uploaded_file) if uploaded_file and uploaded_file.type in ["image/jpeg", "image/png"] else None

    query = st.chat_input("💬 Ask me anything (e.g., 'search', 'think', 'generate image')...")

    if query and not st.session_state.query_processing:
        st.session_state.query_processing = True
        update_query_count(st.session_state.user_id)

        with st.spinner("Processing..."):
            if "search" in query.lower():
                search_results = await perform_google_search(query)
                response = f"Search Results:\n{search_results}"
            elif "think" in query.lower() or "reason" in query.lower():
                response = await query_deepseek(query, st.session_state.user_id)
            elif "generate image" in query.lower():
                response = await generate_image(query)
            else:
                response = await query_gemini(query, st.session_state.user_id, image_data, file_content)

            store_chat(st.session_state.user_id, query, response)
            st.session_state.chat_history.append({"User": query, "AI": response})
            st.session_state.query_processing = False
            st.rerun()

    st.subheader("Chat History")
    for chat in reversed(fetch_chat_history(st.session_state.user_id)):
        with st.chat_message("user"):
            st.markdown(f"**You**: {chat['user']}")
        with st.chat_message("ai"):
            st.markdown(f"**AI**: {chat['ai']}")

# Main Execution
profile_ui()
asyncio.run(chatbot_ui())
