"""Streamlit chat UI for the bilingual TR/EN RAG assistant.

A ChatGPT-style layout with:
  - Left sidebar: brand + topic categories + settings
  - Main: chat bubbles + suggested prompt chips + sticky chat input
  - Academic Calendar topic also shows a FullCalendar widget with university events

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402
from streamlit_calendar import calendar as _calendar  # noqa: E402

from src.config import TOP_K  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402


# ============================================================================
# Calendar helpers
# ============================================================================
TR_MONTHS = {
    "ocak": 1, "şubat": 2, "subat": 2, "mart": 3, "nisan": 4,
    "mayıs": 5, "mayis": 5, "haziran": 6, "temmuz": 7,
    "ağustos": 8, "agustos": 8, "eylül": 9, "eylul": 9,
    "ekim": 10, "kasım": 11, "kasim": 11, "aralık": 12, "aralik": 12,
}


def _parse_tr_date(s: str) -> str | None:
    """Convert ``08 Temmuz 2026`` -> ``2026-07-08``. Return None if unparseable."""
    if not s:
        return None
    parts = s.strip().split()
    if len(parts) != 3:
        return None
    day_s, month_name, year_s = parts
    month = TR_MONTHS.get(month_name.lower())
    if not month:
        return None
    try:
        return f"{int(year_s):04d}-{month:02d}-{int(day_s):02d}"
    except ValueError:
        return None


@st.cache_data(show_spinner=False)
def load_calendar_events() -> list[dict]:
    """Load university events from data/raw/uskudar_events.json into FullCalendar format."""
    events_file = PROJECT_ROOT / "data" / "raw" / "uskudar_events.json"
    if not events_file.exists():
        return []
    raw = json.loads(events_file.read_text(encoding="utf-8"))
    palette = ["#bfe1cb", "#a8dcb6", "#cfe9d6", "#b3e0bf", "#9fd6ad", "#c7ead0"]
    events: list[dict] = []
    for i, item in enumerate(raw):
        date_iso = _parse_tr_date(item.get("date", ""))
        if not date_iso:
            continue
        events.append({
            "title": item.get("title") or "Etkinlik",
            "start": date_iso,
            "end": date_iso,
            "allDay": True,
            "backgroundColor": palette[i % len(palette)],
            "borderColor": palette[i % len(palette)],
            "url": item.get("source_url") or "",
        })
    return events


# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="Smart Student System | Chatbot",
    page_icon="🌻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Theme — CSS
# ============================================================================
LEAF_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">'
    '<path d="M92 18C58 20 31 39 24 70c-4 19 8 34 27 32 33-4 50-38 41-84Z" '
    'fill="#5f9f72" opacity="0.7"/>'
    '<path d="M31 94C46 71 61 51 88 23" fill="none" stroke="#2f6f48" '
    'stroke-width="5" stroke-linecap="round" opacity="0.5"/>'
    "</svg>"
)

THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* hide Streamlit chrome — but KEEP the header at full height so the
   native sidebar-expand toggle stays visible and clickable. */
#MainMenu, footer {{ display: none !important; }}
header[data-testid="stHeader"] {{
    background: transparent !important;
    box-shadow: none !important;
}}
/* Force the collapsed-sidebar control to be visible & styled, no matter
   which testid Streamlit's version uses. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="headerNoPadding"] {{
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 9999 !important;
}}

html, body, [data-testid="stAppViewContainer"] {{
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}}

/* page background: sage-mint gradient */
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at 15% 12%, rgba(197,232,240,0.85), transparent 30rem),
        radial-gradient(circle at 82% 16%, rgba(151,221,190,0.70), transparent 28rem),
        radial-gradient(circle at 48% 88%, rgba(102,176,139,0.40), transparent 30rem),
        linear-gradient(135deg,#eaf6f0 0%,#d4ecde 45%,#bfe1cb 100%) !important;
    min-height: 100vh;
}}
[data-testid="stMain"] {{ background: transparent !important; }}

/* falling leaves */
.leaf-fall-layer {{
    position: fixed; inset: 0; pointer-events: none; overflow: hidden; z-index: 0;
}}
.leaf {{
    position: fixed; top: -10vh; opacity: 0.5;
    animation: leafFall linear infinite;
}}
.leaf-1 {{ left:18%; width:40px; animation-duration:14s; }}
.leaf-2 {{ left:38%; width:34px; animation-duration:17s; animation-delay:2s; }}
.leaf-3 {{ left:62%; width:46px; animation-duration:15s; animation-delay:4s; }}
.leaf-4 {{ left:85%; width:30px; animation-duration:19s; animation-delay:1s; }}
@keyframes leafFall {{
    0%   {{ transform: translateY(-10vh) translateX(0)    rotate(0deg); }}
    50%  {{ transform: translateY(55vh)  translateX(-25px) rotate(180deg); }}
    100% {{ transform: translateY(110vh) translateX(10px) rotate(360deg); }}
}}

/* container width */
.block-container {{
    max-width: 1200px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 8rem !important;
    position: relative; z-index: 5;
}}

/* ---------- Sidebar (light cream, dark green text) ---------- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #f4faf6 0%, #e4f3ea 100%) !important;
    border-right: 1px solid rgba(105,173,130,0.25);
}}
section[data-testid="stSidebar"] * {{ color: #16352f !important; }}
section[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    text-align: left !important;
    background: rgba(255,255,255,0.7) !important;
    border: 1px solid rgba(105,173,130,0.25) !important;
    color: #2f5a3f !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    border-radius: 12px !important;
    min-width: 0 !important;
    box-shadow: none !important;
    transition: background 0.15s, border-color 0.15s;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: #ffffff !important;
    border-color: rgba(105,173,130,0.55) !important;
    color: #16352f !important;
    transform: none !important;
}}
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button.active {{
    background: linear-gradient(135deg,#bfe1cb,#9feeb4) !important;
    border-color: rgba(48,90,63,0.45) !important;
    color: #16352f !important;
}}
.sidebar-brand {{
    display: flex; align-items: center; gap: 10px;
    font-size: 20px; font-weight: 800; color: #16352f;
    padding: 4px 4px 18px; border-bottom: 1px solid rgba(105,173,130,0.25);
    margin-bottom: 14px;
}}
.sidebar-section-label {{
    color: #2f5a3f !important; text-transform: uppercase; letter-spacing: 0.12em;
    font-size: 11px; font-weight: 800;
    margin: 16px 0 6px 4px;
}}

/* ---------- Page title ---------- */
.page-title {{
    font-size: 28px; font-weight: 800; color: #16352f;
    margin: 0 0 4px;
}}
.page-subtitle {{
    color: #2f5a3f; font-weight: 500; margin: 0 0 24px;
}}

/* ---------- Chat messages (st.chat_message) ----------
   We can't reliably detect user vs assistant via :has() across Streamlit
   versions, so all messages get the same readable bubble: white card with
   dark green text. The avatar emoji distinguishes them visually. */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: 0 !important;
    padding: 6px 0 !important;
    margin-bottom: 6px;
}}
/* The bubble itself — match any inner element that holds the markdown.
   Streamlit's exact testid for content varies; cover the common ones. */
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
    background: #ffffff !important;
    border: 1px solid rgba(105,173,130,0.25) !important;
    border-radius: 18px !important;
    padding: 14px 18px !important;
    color: #16352f !important;
    box-shadow: 0 6px 20px rgba(24,52,42,0.06);
}}
/* Force every text node inside any chat message to dark green. */
[data-testid="stChatMessage"],
[data-testid="stChatMessage"] *,
[data-testid="stChatMessage"] .stMarkdown,
[data-testid="stChatMessage"] .stMarkdown *,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {{
    color: #16352f !important;
}}

/* ---------- Suggested prompt chips ---------- */
.chip-row {{ margin-top: 8px; }}
[data-testid="stHorizontalBlock"] .stButton > button[data-testid^="stBaseButton"] {{
    /* leave existing rules alone; we override below for chip buttons */
}}
.chip-grid .stButton > button {{
    width: 100%;
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid rgba(105,173,130,0.30) !important;
    color: #2f5a3f !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 12px 16px !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 14px rgba(24,52,42,0.06) !important;
    min-height: 56px;
    white-space: normal !important;
    line-height: 1.35 !important;
    transition: transform 0.12s, border-color 0.15s, background 0.15s;
}}
.chip-grid .stButton > button:hover {{
    transform: translateY(-1px);
    background: #ffffff !important;
    border-color: rgba(105,173,130,0.55) !important;
    color: #16352f !important;
    box-shadow: 0 8px 22px rgba(24,52,42,0.10) !important;
}}

/* ---------- Chat input (sticky bottom, rounded ChatGPT-style) ---------- */
/* Aggressively make the entire bottom-area chrome transparent — Streamlit
   renders a dark wrapper around stChatInput that we need to hide.
   We target the testids AND nuke the background on every descendant div so
   any inner wrapper Streamlit adds doesn't bring the black bar back. */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > div,
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
div[class*="stBottom"],
div[class*="BottomBlock"],
div[class*="stChatInput"] {{
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    border-top: 0 !important;
    box-shadow: none !important;
    max-width: 100% !important;
    width: 100% !important;
}}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {{
    max-width: 1200px !important;
    width: 100% !important;
    margin: 0 auto !important;
}}
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] [data-testid="stChatInputContainer"] {{
    width: 100% !important;
    max-width: 100% !important;
}}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] [data-testid="stChatInputTextArea"] {{
    background: #ffffff !important;
    border: 1px solid rgba(105,173,130,0.35) !important;
    border-radius: 22px !important;
    padding: 14px 56px 14px 22px !important;
    color: #16352f !important;
    font-size: 15px !important;
    box-shadow: 0 10px 30px rgba(24,52,42,0.10) !important;
    width: 100% !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: #345039 !important;
    box-shadow: 0 10px 30px rgba(24,52,42,0.15),
                0 0 0 3px rgba(159,238,180,0.30) !important;
    outline: none !important;
}}
[data-testid="stChatInputSubmitButton"] {{
    background: linear-gradient(135deg,#bfe1cb 0%,#9feeb4 100%) !important;
    border: 1px solid rgba(48,90,63,0.30) !important;
    border-radius: 14px !important;
    color: #16352f !important;
}}
[data-testid="stChatInputSubmitButton"] svg {{ color: #16352f !important; fill: #16352f !important; }}

/* ---------- Language picker pill row ---------- */
.lang-pills {{
    display: flex; gap: 8px; justify-content: center;
    margin: 0 auto 8px; max-width: 880px;
}}
.lang-pills .stButton > button {{
    background: rgba(255,255,255,0.7) !important;
    border: 1px solid rgba(105,173,130,0.30) !important;
    color: #2f5a3f !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
    border-radius: 999px !important;
    min-width: 0 !important;
    box-shadow: none !important;
    min-height: 0 !important;
}}
.lang-pills .stButton > button:hover {{
    background: #ffffff !important;
    color: #16352f !important;
    transform: none !important;
}}

/* Text inside main */
[data-testid="stMain"] :is(p, span, li, strong, em, h1, h2, h3, h4, h5, h6),
[data-testid="stMain"] .stMarkdown,
[data-testid="stMain"] .stMarkdown * {{
    color: #16352f;
}}
section[data-testid="stSidebar"] * {{ color: #16352f !important; }}
section[data-testid="stSidebar"] .sidebar-section-label {{ color: #2f5a3f !important; }}

/* Calendar: dark green text on light backgrounds */
.fc-col-header-cell {{ background: #d4ecde !important; }}
.fc-col-header-cell-cushion {{ color: #16352f !important; }}
.fc-button, .fc-button-primary {{ color: #16352f !important; }}
.fc-button:hover {{ color: #16352f !important; }}
/* Calendar event chips: tint backgrounds so dark text is readable */
.fc-event, .fc-event * {{ color: #16352f !important; }}
</style>
"""

LEAVES_HTML = (
    '<div class="leaf-fall-layer">'
    + "".join(f'<div class="leaf leaf-{i}">{LEAF_SVG}</div>' for i in range(1, 5))
    + "</div>"
)

SIDEBAR_TOGGLE_HTML = """
<style>
.custom-sidebar-toggle {
    position: fixed;
    top: 14px;
    left: 14px;
    z-index: 99999;
    background: linear-gradient(135deg, #bfe1cb 0%, #9feeb4 100%);
    color: #16352f;
    border: 1px solid rgba(48,90,63,0.30);
    border-radius: 12px;
    padding: 8px 14px;
    font-family: Inter, sans-serif;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    box-shadow: 0 8px 22px rgba(24,52,42,0.18);
    transition: transform 0.12s, box-shadow 0.15s;
    user-select: none;
}
.custom-sidebar-toggle:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(24,52,42,0.22);
}
/* Hide our custom button when the sidebar is already expanded
   (Streamlit sets aria-expanded on the sidebar). */
section[data-testid="stSidebar"][aria-expanded="true"] ~ * .custom-sidebar-toggle,
body:has(section[data-testid="stSidebar"][aria-expanded="true"]) .custom-sidebar-toggle {
    display: none;
}
</style>
<div class="custom-sidebar-toggle" onclick="
    const candidates = [
        '[data-testid=\\'stSidebarCollapsedControl\\'] button',
        '[data-testid=\\'collapsedControl\\'] button',
        '[data-testid=\\'stSidebarCollapseButton\\']',
        '[aria-label*=\\'sidebar\\' i]',
        '[aria-label*=\\'menu\\' i]'
    ];
    for (const sel of candidates) {
        const el = document.querySelector(sel);
        if (el) { el.click(); break; }
    }
">☰ Menu</div>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(LEAVES_HTML, unsafe_allow_html=True)
st.markdown(SIDEBAR_TOGGLE_HTML, unsafe_allow_html=True)


# ============================================================================
# Categories
# ============================================================================
CATEGORIES = {
    "General Questions": {
        "icon": "💬",
        "hint": "",
        "subtitle_en": "Ask anything about Üsküdar University.",
        "subtitle_tr": "Üsküdar Üniversitesi hakkında her şeyi sor.",
        "suggestions": [
            "What sections should my graduation report include?",
            "Where is the cafeteria?",
            "Kütüphane hafta sonu açık mı?",
            "Kampüste mescit var mı?",
        ],
    },
    "Erasmus": {
        "icon": "🌍",
        "hint": "erasmus değişim programı",
        "subtitle_en": "Erasmus exchange program — applications, mobility, deadlines.",
        "subtitle_tr": "Erasmus değişim programı — başvuru, hareketlilik, son tarihler.",
        "suggestions": [
            "How can I apply for Erasmus?",
            "Erasmus için son başvuru tarihi nedir?",
            "Which countries can I go to?",
            "Erasmus stajı nasıl yapılır?",
        ],
    },
    "Graduation Project": {
        "icon": "🎓",
        "hint": "mezuniyet projesi graduation report",
        "subtitle_en": "Graduation project rules, report sections, deadlines.",
        "subtitle_tr": "Mezuniyet projesi kuralları, bölümleri ve tarihleri.",
        "suggestions": [
            "What sections should my graduation report include?",
            "Mezuniyet projesi raporunda neler olmalı?",
            "Project submission deadline?",
            "Bitirme savunma tarihleri ne zaman?",
        ],
    },
    "Campus Life": {
        "icon": "🏫",
        "hint": "kampüs yaşamı kulüp öğrenci kafeterya",
        "subtitle_en": "Campus, clubs, dining, facilities.",
        "subtitle_tr": "Kampüs, kulüpler, yemekhane, tesisler.",
        "suggestions": [
            "What student clubs are there?",
            "Kafeterya nerede?",
            "Spor salonu hangi blokta?",
            "Kütüphane çalışma saatleri?",
        ],
    },
    "Announcements": {
        "icon": "📢",
        "hint": "duyuru announcement",
        "subtitle_en": "Latest university announcements.",
        "subtitle_tr": "Üniversitenin son duyuruları.",
        "suggestions": [
            "Ders kayıt duyurusu var mı?",
            "When are the registration dates?",
            "Akademik kadro ilanları",
            "Important deadlines this semester",
        ],
    },
    "Academic Calendar": {
        "icon": "📅",
        "hint": "akademik takvim sınav tarih exam date",
        "subtitle_en": "Exam dates, registration windows, semester schedule.",
        "subtitle_tr": "Sınav tarihleri, kayıt dönemleri, akademik takvim.",
        "suggestions": [
            "Where can I find exam dates?",
            "Vize sınavları ne zaman?",
            "Final sınav takvimi",
            "Ders kayıt tarihleri",
        ],
    },
}


# ============================================================================
# Session state
# ============================================================================
def _seed_welcome() -> dict:
    return {
        "role": "assistant",
        "content": (
            "👋 Merhaba, ben Üsküdar Üniversitesi öğrenci rehberinim. / "
            "Hi, I'm your Üsküdar University guide.\n\n"
            "Pick a topic from the sidebar or just ask me anything in **Turkish or English**. "
            "Sol menüden bir konu seç veya doğrudan **Türkçe ya da İngilizce** sorunu yaz."
        ),
    }


if "messages" not in st.session_state:
    st.session_state.messages = [_seed_welcome()]
if "category" not in st.session_state:
    st.session_state.category = "General Questions"
if "language_override" not in st.session_state:
    st.session_state.language_override = None  # None = auto
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================================
# Pipeline (cached, pre-warmed)
# ============================================================================
@st.cache_resource(show_spinner="Loading retriever and generator (first call only)...")
def get_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline()
    try:
        pipeline.answer("warmup", top_k=1)
    except Exception:
        pass
    return pipeline


get_pipeline()  # eager warmup


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">🌻 Smart Student System</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">Topics</div>', unsafe_allow_html=True)
    for cat_name, info in CATEGORIES.items():
        is_active = cat_name == st.session_state.category
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            f"{info['icon']}  {cat_name}",
            key=f"cat_btn_{cat_name}",
            type=btn_type,
            use_container_width=True,
        ):
            st.session_state.category = cat_name
            # Reset to welcome message when switching topic
            st.session_state.messages = [_seed_welcome()]
            st.rerun()

    st.markdown('<div class="sidebar-section-label">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Context chunks", min_value=1, max_value=10, value=TOP_K)

    if st.button("🗑️ Clear chat", use_container_width=True, key="clear_chat"):
        st.session_state.messages = [_seed_welcome()]
        st.rerun()


# ============================================================================
# Main area: header
# ============================================================================
current_cat = st.session_state.category
cat_info = CATEGORIES[current_cat]

st.markdown(
    f'<h1 class="page-title">{cat_info["icon"]} {current_cat}</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="page-subtitle">{cat_info["subtitle_en"]}<br>{cat_info["subtitle_tr"]}</p>',
    unsafe_allow_html=True,
)


# ============================================================================
# Academic Calendar topic: show a FullCalendar widget above the chat
# ============================================================================
if current_cat == "Academic Calendar":
    cal_events = load_calendar_events()
    cal_options = {
        "initialView": "dayGridMonth",
        "headerToolbar": {
            "left": "prev,next today",
            "center": "title",
            "right": "dayGridMonth,timeGridWeek,listMonth",
        },
        "locale": "tr",
        "firstDay": 1,  # Monday
        "height": 620,
        "buttonText": {
            "today": "Bugün",
            "month": "Ay",
            "week": "Hafta",
            "list": "Liste",
        },
        "dayMaxEvents": 3,
    }
    _calendar(
        events=cal_events,
        options=cal_options,
        custom_css="""
            .fc-toolbar-title { color: #16352f; font-weight: 800; }
            .fc-button, .fc-button-primary {
                background: linear-gradient(135deg,#d4ecde,#bfe1cb) !important;
                border: 1px solid rgba(48,90,63,0.30) !important;
                color: #16352f !important; font-weight: 700 !important;
            }
            .fc-button:hover {
                background: linear-gradient(135deg,#bfe1cb,#9feeb4) !important;
                color: #16352f !important;
            }
            .fc-daygrid-day-number { color: #16352f !important; font-weight: 600; }
            .fc-day-today { background: rgba(159,238,180,0.25) !important; }
            .fc-col-header-cell { background: #d4ecde !important; }
            .fc-col-header-cell-cushion { color: #16352f !important; font-weight: 700; }
            .fc-event, .fc-event * { color: #16352f !important; }
            .fc { background: rgba(255,255,255,0.92); border-radius: 18px;
                  padding: 14px; box-shadow: 0 12px 32px rgba(24,52,42,0.08); }
        """,
        key="academic_calendar",
    )
    st.caption(
        f"📌 Loaded **{len(cal_events)}** events from `data/raw/uskudar_events.json`. "
        "Click an event to open its source page."
    )
    st.divider()


# ============================================================================
# Chat history
# ============================================================================
for msg in st.session_state.messages:
    avatar = "🌻" if msg["role"] == "assistant" else "🧑‍🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ============================================================================
# Suggested prompt chips (only when conversation is just the welcome)
# ============================================================================
if len(st.session_state.messages) == 1:
    st.markdown('<div class="chip-grid">', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, prompt in enumerate(cat_info["suggestions"]):
        with cols[i % 2]:
            if st.button(prompt, key=f"chip_{current_cat}_{i}"):
                st.session_state.pending_prompt = prompt
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# Language pills (Auto / English / Türkçe) — above the chat input
# ============================================================================
st.markdown('<div class="lang-pills">', unsafe_allow_html=True)
lp1, lp2, lp3, lp_spacer = st.columns([1, 1, 1, 6])
LANG_LABELS = [("Auto", None), ("English", "en"), ("Türkçe", "tr")]
for col, (label, code) in zip([lp1, lp2, lp3], LANG_LABELS):
    is_sel = st.session_state.language_override == code
    with col:
        if st.button(
            ("● " if is_sel else "○ ") + label,
            key=f"lang_pill_{label}",
            use_container_width=True,
        ):
            st.session_state.language_override = code
            st.rerun()
st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# Sticky chat input
# ============================================================================
typed = st.chat_input(
    placeholder="Ask anything... / Sorunuzu yazın..."
)

# Either an inline-typed question or a clicked suggestion chip kicks off generation.
question = typed or st.session_state.pending_prompt
if st.session_state.pending_prompt:
    st.session_state.pending_prompt = None

if question:
    # 1. Echo user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    # 2. Run RAG — augment query with category hint for better retrieval
    augmented = (cat_info["hint"] + " " + question).strip() if cat_info["hint"] else question
    pipeline = get_pipeline()
    with st.chat_message("assistant", avatar="🌻"):
        with st.spinner("Thinking..."):
            try:
                result = pipeline.answer(
                    augmented,
                    top_k=top_k,
                    language_override=st.session_state.language_override,
                )
                answer_text = result.answer or "_(empty answer)_"
            except Exception as e:
                answer_text = f"_(error: {e})_"
                result = None
        st.markdown(answer_text)

        # Show retrieved context as a small expander
        if result and result.retrieved_chunks:
            with st.expander(f"📚 Sources ({len(result.retrieved_chunks)})"):
                for i, chunk in enumerate(result.retrieved_chunks, start=1):
                    st.markdown(
                        f"**[{i}] {chunk.get('title') or '(untitled)'}**  "
                        f"`{chunk.get('source','')}`  · score `{chunk.get('score', 0.0):.2f}`"
                    )
                    st.caption(chunk["text"][:400] + ("…" if len(chunk["text"]) > 400 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    st.rerun()
