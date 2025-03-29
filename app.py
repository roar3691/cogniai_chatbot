"""
CogniChat: A Streamlit-based chatbot with OpenRouter LLM integration.
Uses MongoDB to store chats. Optimized for Hugging Face Spaces deployment.
"""

import subprocess
import sys
import asyncio
import aiohttp
from collections import deque
from datetime import datetime, timedelta
import uuid
import random
import re
import os
import streamlit as st

# Dynamic Package Installation (from reference)
def install_package(package):
    try:
        __import__(package.split("==")[0].replace("-", "_"))
    except ImportError:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            sys.exit(1)

required_packages = [
    "streamlit==1.27.0",
    "pymongo==4.5.0",
    "aiohttp==3.8.5",
    "bcrypt==4.0.1",
    "google-api-python-client==2.100.0",
    "PyPDF2==3.0.1"
]

for package in required_packages:
    install_package(package)

# Imports after installation
from pymongo import MongoClient, ASCENDING
from googleapiclient.discovery import build
import PyPDF2
import bcrypt

# Load environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Replace or set in secrets
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB Setup (aligned with reference)
client = MongoClient(MONGO_URI)
db = client["cognichat_db"]  # Match reference DB name
chat_collection = db["chats"]  # Match reference collection name
profile_collection = db["profiles"]
MAX_CHAT_HISTORY = 500
chat_collection.create_index([("user_id", ASCENDING), ("conversation_id", ASCENDING), ("timestamp", ASCENDING)])

# Session State Initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = deque(maxlen=50)
if "query_processing" not in st.session_state:
    st.session_state.query_processing = False
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {"tone": "formal", "detail_level": "medium", "language": "en", "format": "paragraph"}
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "last_proactive_time" not in st.session_state:
    st.session_state.last_proactive_time = 0
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0

# Helper Functions
def heuristic_multi_scale_attention(query):
    words = query.lower().split()
    length = len(words)
    short_scale = min(1.0, length / 5)
    specific_keywords = {"what", "how", "why", "who", "where", "when", "explain", "describe"}
    mid_scale = sum(1 for word in words if word in specific_keywords) / max(1, length)
    long_scale = 1.0 if "?" in query or len(re.findall(r"\w+", query)) > 3 else 0.5
    return min(max(0.3 * short_scale + 0.4 * mid_scale + 0.3 * long_scale, 0.1), 1.0)

def adjust_focus_score(query, focus_score):
    words = query.lower().split()
    if any(word in words for word in ["what", "how", "why", "who", "where", "when"]):
        act_bonus = 0.2
        if any(word in words for word in ["great", "good", "happy", "cool"]):
            emotion_bonus = 0.1
        else:
            emotion_bonus = 0.0
    elif any(word in words for word in ["tell", "give", "show"]):
        act_bonus = 0.1
        emotion_bonus = 0.0
    else:
        act_bonus = -0.1
        emotion_bonus = 0.0 if "please" in words else -0.1
    return min(max(focus_score + act_bonus + emotion_bonus, 0.1), 1.0)

# Profile Management Functions (with bcrypt from reference)
def create_profile(username, password):
    user_id = str(uuid.uuid4())
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    profile = {
        "user_id": user_id,
        "username": username,
        "password": hashed_password,
        "preferences": {"tone": "formal", "detail_level": "medium", "language": "en", "format": "paragraph"},
        "created_at": datetime.utcnow().timestamp(),
        "interests": {},
        "query_count": 0,
        "last_query_time": 0
    }
    profile_collection.insert_one(profile)
    return user_id

def authenticate_user(username, password):
    user = profile_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        prefs = user.get("preferences", {})
        updates = {}
        if "language" not in prefs:
            prefs["language"] = "en"
            updates["preferences"] = prefs
        if "format" not in prefs:
            prefs["format"] = "paragraph"
            updates["preferences"] = prefs
        if "last_query_time" not in user:
            updates["last_query_time"] = 0
        if updates:
            profile_collection.update_one({"user_id": user["user_id"]}, {"$set": updates})
        return user["user_id"]
    return None

def get_user_preferences(user_id):
    user = profile_collection.find_one({"user_id": user_id})
    if user:
        prefs = user.get("preferences", {"tone": "formal", "detail_level": "medium", "language": "en", "format": "paragraph"})
        updates = {}
        if "language" not in prefs:
            prefs["language"] = "en"
            updates["preferences"] = prefs
        if "format" not in prefs:
            prefs["format"] = "paragraph"
            updates["preferences"] = prefs
        if "last_query_time" not in user:
            updates["last_query_time"] = 0
        if updates:
            profile_collection.update_one({"user_id": user_id}, {"$set": updates})
        return prefs
    return {"tone": "formal", "detail_level": "medium", "language": "en", "format": "paragraph"}

def update_user_preferences(user_id, preferences):
    profile_collection.update_one({"user_id": user_id}, {"$set": {"preferences": preferences}})

def update_user_interests(user_id, interests):
    profile_collection.update_one({"user_id": user_id}, {"$set": {"interests": interests}})

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

# Chat History Management (with conversation_id from reference)
def fetch_chat_history(user_id, conversation_id=None, limit=5):
    query = {"user_id": user_id}
    if conversation_id:
        query["conversation_id"] = conversation_id
    return list(chat_collection.find(query, {"_id": 0, "user_message": 1, "ai_response": 1, "rating": 1, "timestamp": 1})
                .sort("timestamp", -1).limit(limit))

def store_chat(user_id, conversation_id, query, response, rating=None):
    chat_entry = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "user_message": query,
        "ai_response": response,
        "timestamp": datetime.utcnow().timestamp(),
        "rating": rating
    }
    chat_collection.insert_one(chat_entry)
    if chat_collection.count_documents({"user_id": user_id}) > MAX_CHAT_HISTORY:
        oldest = chat_collection.find_one({"user_id": user_id}, sort=[("timestamp", ASCENDING)])
        chat_collection.delete_one({"_id": oldest["_id"]})

# Conversation Summarization
def summarize_history(user_id, conversation_id=None):
    history = fetch_chat_history(user_id, conversation_id, 20)
    if not history:
        return "No recent conversation to summarize."
    summary = "Recent chat summary:\n"
    for chat in history:
        summary += f"- You asked: '{chat['user_message'][:50]}...', I replied: '{chat['ai_response'][:50]}...'\n"
    return summary.strip()

# Detect User Interests
def detect_user_interests(user_id):
    history = fetch_chat_history(user_id, limit=20)
    interests = {}
    for chat in history:
        words = chat["user_message"].lower().split()
        for word in words:
            if len(word) > 3:
                interests[word] = interests.get(word, 0) + 1
    return dict(sorted(interests.items(), key=lambda x: x[1], reverse=True)[:3])

# Proactive Suggestion
def generate_proactive_suggestion(user_id):
    interests = detect_user_interests(user_id)
    if interests and random.random() > 0.3:
        top_interest = max(interests, key=interests.get)
        return f"Hey, noticed you’re into {top_interest}. Want to chat about it?"
    topics = ["latest news", "fun trivia", "math puzzles"]
    return f"How about we discuss {random.choice(topics)}?"

# Multi-Modal Processing (PDF Only)
def process_uploaded_file(file):
    if file.type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return f"Extracted text from PDF: {text}" if text else "No text could be extracted from the PDF."
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    return "Unsupported file type (only PDFs are supported)."

# AI Query Function using OpenRouter (adapted from reference)
async def query_llm(query, user_id, file_content=None):
    preferences = get_user_preferences(user_id)
    focus_score = heuristic_multi_scale_attention(query)
    focus_score = adjust_focus_score(query, focus_score)
    
    past_context = summarize_history(user_id, st.session_state.conversation_id)
    google_results = await perform_google_search(query) if not file_content else "N/A"
    
    if file_content:
        query = f"{query}\n\nFile Content: {file_content}"

    prompt = f"""
    **User Query**: "{query}"
    **Contextual History**: 
    {past_context}
    **Google Search Results**: 
    {google_results}
    **User Preferences**: Tone: {preferences['tone']}, Detail Level: {preferences['detail_level']}, Language: {preferences['language']}, Format: {preferences['format']}
    **Focus Score**: {focus_score:.2f}
    **Instructions**:
    - Respond in a {preferences['tone']} tone with {preferences['detail_level']} detail in {preferences['language']}.
    - Format as {preferences['format']} (e.g., paragraphs or bullet points).
    - Use contextual history for personalization; summarize if long.
    - Incorporate file content or external API data if relevant.
    - Keep greetings engaging; ensure questions are answered accurately.
    - If unclear, ask for clarification politely.
    """

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://roar3691-cognichat.hf.space",
                    "X-Title": "CogniChat"
                },
                json={
                    "model": "deepseek/deepseek-chat-v3-0324:free",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.9
                }
            ) as response:
                result = await response.json()
                if "choices" not in result:
                    st.error(f"Invalid LLM response: {result}")
                    return "Sorry, I couldn’t process that. Please try again."
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            st.error(f"❌ OpenRouter AI Error: {e}")
            return f"Sorry, something went wrong! How can I assist with '{query}'?"

# Profile UI
def profile_ui():
    st.set_page_config(page_title="CogniChat", page_icon="💬")
    st.title("CogniChat")
    
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
                st.session_state.last_query_time = profile_collection.find_one({"user_id": user_id}).get("last_query_time", 0)
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials!")
    else:
        st.subheader(f"Welcome, User {st.session_state.user_id[:8]}!")
        with st.expander("Preferences"):
            tone = st.selectbox("Tone", ["formal", "casual"], index=["formal", "casual"].index(st.session_state.user_preferences["tone"]))
            detail = st.selectbox("Detail Level", ["low", "medium", "high"], index=["low", "medium", "high"].index(st.session_state.user_preferences["detail_level"]))
            current_lang = st.session_state.user_preferences.get("language", "en")
            lang = st.selectbox("Language", ["en", "es", "fr", "hi"], index=["en", "es", "fr", "hi"].index(current_lang))
            current_format = st.session_state.user_preferences.get("format", "paragraph")
            format = st.selectbox("Format", ["paragraph", "bullet"], index=["paragraph", "bullet"].index(current_format))
            if st.button("Update Preferences"):
                new_prefs = {"tone": tone, "detail_level": detail, "language": lang, "format": format}
                update_user_preferences(st.session_state.user_id, new_prefs)
                st.session_state.user_preferences = new_prefs
                st.success("Preferences updated!")
        if st.button("Logout"):
            st.session_state.user_id = None
            st.session_state.conversation_id = None
            st.session_state.chat_history.clear()
            st.session_state.last_query_time = 0
            st.rerun()

# Analytics Dashboard
def analytics_ui(user_id):
    with st.expander("Analytics Dashboard"):
        history = fetch_chat_history(user_id, limit=50)
        if history:
            st.write(f"Total Interactions: {len(history)}")
            interests = detect_user_interests(user_id)
            st.write("Top Interests:", ", ".join([f"{k} ({v})" for k, v in interests.items()]))
            ratings = [chat.get("rating", 3) for chat in history if chat.get("rating")]
            st.write(f"Average Rating: {sum(ratings)/len(ratings):.2f}" if ratings else "No ratings yet.")

# Chatbot UI
def chatbot_ui():
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
    if user:
        current_time = datetime.utcnow().timestamp()
        last_query_time = user.get("last_query_time", 0)
        if current_time - last_query_time < 2:
            st.warning("Please wait a moment before sending another query.")
            return
        if user.get("query_count", 0) > 50:
            st.error("Query limit reached for today.")
            return

    if current_time - st.session_state.last_proactive_time > 30 and not st.session_state.query_processing:
        async def proactive_suggestion():
            suggestion = await query_llm(generate_proactive_suggestion(st.session_state.user_id), st.session_state.user_id)
            st.session_state.notifications.append(f"AI Suggests: {suggestion}")
            st.session_state.last_proactive_time = current_time
            st.rerun()
        asyncio.run(proactive_suggestion())

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    file_content = process_uploaded_file(uploaded_file) if uploaded_file else None

    query = st.chat_input("💬 Type your message...") or st.session_state.last_query
    
    if query and not st.session_state.query_processing:
        if query != st.session_state.last_query or file_content:
            st.session_state.query_processing = True
            st.session_state.last_query = query
            st.session_state.last_response = None
            if not st.session_state.conversation_id:
                st.session_state.conversation_id = str(uuid.uuid4())
            update_query_count(st.session_state.user_id)

            with st.spinner("Processing your query..."):
                ai_response = asyncio.run(query_llm(query, st.session_state.user_id, file_content))
                store_chat(st.session_state.user_id, st.session_state.conversation_id, query, ai_response)
                st.session_state.last_response = ai_response
                st.session_state.chat_history.append({"User": query, "AI": ai_response})
                st.session_state.query_processing = False
                st.rerun()

    st.subheader("Chat History")
    for i, chat in enumerate(reversed(fetch_chat_history(st.session_state.user_id, st.session_state.conversation_id, 20))):
        with st.chat_message("user"):
            st.markdown(f"**You**: {chat['user_message']}")
        with st.chat_message("ai"):
            st.markdown(f"**AI**: {chat['ai_response']}")
            rating = st.slider(f"Rate this", 1, 5, chat.get("rating", 3), key=f"rating_{i}_{chat['timestamp']}")
            if st.button("Submit Rating", key=f"submit_{i}_{chat['timestamp']}"):
                chat_collection.update_one(
                    {"user_id": st.session_state.user_id, "conversation_id": st.session_state.conversation_id, "timestamp": chat["timestamp"]},
                    {"$set": {"rating": rating}}
                )
                st.success(f"Rating {rating} submitted!")
        st.markdown("---")

    analytics_ui(st.session_state.user_id)

# Main App Layout
profile_ui()
chatbot_ui()
