import streamlit as st
import requests
from rag_utils import query_rag  # RAG utilities

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Construction AI Assistant",
    page_icon="🏗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- CSS Styling -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root {
    --primary-bg: #F5F2E7; --secondary-bg: #E8E4D9; --accent-green: #6B8E23;
    --accent-orange: #D2691E; --clay-brown: #8B5E3C; --text-dark: #3B3B3B; --text-light: #5C4033;
}
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}
.stApp {background: var(--primary-bg) !important; font-family: 'Inter', sans-serif !important; color: var(--text-dark) !important;}
.main-container {max-width: 850px; margin: 30px auto; background: var(--secondary-bg); border-radius: 20px; padding: 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15); overflow: hidden; position: relative;}
.chat-header {background: var(--accent-green); color: white; padding: 25px; text-align: center; border-bottom: 3px solid var(--clay-brown); position: relative;}
.header-title {font-size: 2rem; font-weight: 700;}
.header-desc {font-size: 1rem; opacity: 0.9; margin-top: 5px;}
.status-indicator {position: absolute; top: 25px; right: 25px; display: flex; align-items: center; gap: 8px; font-size: 14px; background: var(--accent-orange); color: white; padding: 6px 14px; border-radius: 20px;}
.status-dot {width: 10px; height: 10px; background: #00ff00; border-radius: 50%; animation: pulse 1.5s infinite;}
@keyframes pulse {0% { transform: scale(1); opacity: 1; }50% { transform: scale(1.3); opacity: 0.6; }100% { transform: scale(1); opacity: 1; }}
.message-container {padding: 20px;}
.message {display: flex; align-items: flex-start; gap: 12px; margin-bottom: 18px; max-width: 80%;}
.message.user {margin-left: auto; flex-direction: row-reverse;}
.message-avatar {width: 42px; height: 42px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: bold; color: white;}
.message-avatar.user { background: var(--accent-green); }
.message-avatar.bot { background: var(--clay-brown); }
.message-content {padding: 14px 20px; border-radius: 14px; font-size: 15px; line-height: 1.6; border: 1px solid var(--clay-brown);}
.message-content.user {background: var(--accent-green); color: white;}
.message-content.bot {background: var(--secondary-bg); color: var(--text-dark);}
.typing-indicator {display: flex; align-items: center; gap: 10px; padding: 12px 18px; margin: 10px 0; border-radius: 14px; background: var(--secondary-bg); font-size: 14px; font-style: italic; color: var(--text-light);}
.typing-dots {display: flex; gap: 4px;}
.typing-dot {width: 6px; height: 6px; background: var(--text-light); border-radius: 50%; animation: blink 1.2s infinite;}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {0%, 80%, 100% { opacity: 0.3; } 40% { opacity: 1; }}
.input-container {padding: 20px; border-top: 2px solid var(--clay-brown); background: var(--secondary-bg);}
.stTextInput > div > div > input {background: white !important; border: 1px solid var(--clay-brown) !important; border-radius: 12px !important; padding: 14px 20px !important; color: var(--text-dark) !important; font-size: 15px !important;}
.stTextInput > div > div > input::placeholder {color: var(--text-light) !important;}
.stButton > button {background: var(--accent-orange) !important; border-radius: 12px !important; border: none !important; color: white !important; padding: 12px 22px !important; font-weight: 600 !important; transition: all 0.2s ease !important;}
.stButton > button:hover {background: var(--clay-brown) !important;}
.bg-particles {position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; z-index: -1;}
.particle {position: absolute; bottom: -20px; background: rgba(107, 142, 35, 0.3); border-radius: 50%; animation: rise 20s linear infinite;}
@keyframes rise {0% { transform: translateY(0); opacity: 1; } 100% { transform: translateY(-120vh); opacity: 0; }} #MainMenu, footer, header {visibility: hidden;}
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}
.stAppDeployButton {display: none;}
[data-testid="stStatusWidget"] {display: none !important;}  /* Hides "Manage app" */
</style>
""", unsafe_allow_html=True)

# ----------------- Background Particles -----------------
st.markdown("""
<div class="bg-particles">
    <div class="particle" style="left: 10%; width: 5px; height: 5px; animation-delay: 0s;"></div>
    <div class="particle" style="left: 30%; width: 8px; height: 8px; animation-delay: 4s;"></div>
    <div class="particle" style="left: 50%; width: 4px; height: 4px; animation-delay: 8s;"></div>
    <div class="particle" style="left: 70%; width: 7px; height: 7px; animation-delay: 12s;"></div>
    <div class="particle" style="left: 90%; width: 6px; height: 6px; animation-delay: 16s;"></div>
</div>
""", unsafe_allow_html=True)

# ----------------- Main Container -----------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# ----------------- Header -----------------
st.markdown('''
<div class="chat-header">
    <div class="header-title">🏗 Construction AI Assistant</div>
    <div class="header-desc">Shaping the Future of Construction with Intelligent Assistance</div>
    <div class="status-indicator">
        <div class="status-dot"></div>
        <span style="font-size: 14px; font-weight: 500;">Online</span>
    </div>
</div>
''', unsafe_allow_html=True)

# ----------------- Gemini API -----------------
GEMINI_API_KEY = "AIzaSyAdii0tN49b5IF2XrYQ42nSn70nE4av8QA"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# ----------------- Construction Keywords -----------------
CONSTRUCTION_KEYWORDS = [
    "concrete","cement","brick","steel","construction","masonry","plaster",
    "foundation","reinforcement","aggregate","mortar","beam","column","slab",
    "wall","roof","excavation","compaction","curing","mix","grade","strength",
    "load","structural","building","architecture","engineering","materials",
    "tools","equipment","safety","site","project","design","blueprint",
    "scaffold","formwork","rebar","welding","paint","tiles","flooring",
    "plumbing","electrical","hvac","insulation","waterproof","contractor",
    "supervisor","worker","labor","cost","estimate","measurement","quantity",
    "surveying","soil","geotechnical"
]

def is_construction_query(query):
    return any(keyword in query.lower() for keyword in CONSTRUCTION_KEYWORDS)

# ----------------- RAG + Gemini Response -----------------
def get_response(user_query):
    # Step 1: Query RAG for context
    context_chunks = query_rag(user_query, index_path="faiss_index", k=3)

    # Step 2: Prepare the system prompt for Gemini
    if context_chunks:
        context_text = "\n\n".join(context_chunks)
        system_prompt = (
            "You are a construction and civil engineering expert. "
            "Use the context below to answer the user's question. "
            "Expand on the information, provide details, and explain concepts clearly. "
            "If the context does not fully answer the question, supplement your response with your own knowledge.\n\n"
            f"Context:\n{context_text}\n\nUser question: {user_query}"
        )
    else:
        # No context → direct response
        system_prompt = (
            "You are a construction and civil engineering expert. "
            "Answer the user's question fully and clearly with detailed explanations and proper markdown formatting.\n\n"
            f"User question: {user_query}"
        )

    # Step 3: Call Gemini API
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    payload = {"contents": [{"parts": [{"text": system_prompt}]}]}

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response generated.")
    except Exception as e:
        return f"⚠️ Error: {str(e)}"



# ----------------- Session State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "typing" not in st.session_state:
    st.session_state.typing = False

# ----------------- Display Messages -----------------
st.markdown('<div class="message-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    role_class = "user" if message["role"] == "user" else "bot"
    avatar = "👤" if message["role"] == "user" else "🤖"
    st.markdown(f'''
    <div class="message {role_class}">
        <div class="message-avatar {role_class}">{avatar}</div>
        <div class="message-content {role_class}">{message["content"]}</div>
    </div>
    ''', unsafe_allow_html=True)
if st.session_state.typing:
    st.markdown('''
    <div class="typing-indicator">
        <div class="message-avatar bot">🤖</div>
        <span>AI Assistant is analyzing your query</span>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Input Form -----------------
st.markdown('<div class="input-container">', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Your question", 
            placeholder="Ask me about construction techniques, materials, engineering principles...",
            key="user_input", label_visibility="collapsed"
        )
    with col2:
        submit_button = st.form_submit_button("Send 📤", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Process Query -----------------
if submit_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.typing = True
    st.rerun()

if st.session_state.typing and st.session_state.messages:
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user":
        if is_construction_query(last_message["content"]):
            with st.spinner("AI Assistant is analyzing your query..."):
                response = get_response(last_message["content"])
        else:
            response = """🏗 I specialize in *construction and civil engineering* questions only.

Please ask about:
- Concrete technology  
- Steel engineering  
- Building materials  
- Construction methods & equipment  
- Structural design principles  
- Foundation systems"""
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.typing = False
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Auto Scroll -----------------
st.markdown("""
<script>
function scrollToBottom() { window.scrollTo(0, document.body.scrollHeight); }
setTimeout(scrollToBottom, 100);
</script>
""", unsafe_allow_html=True)
