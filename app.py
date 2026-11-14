
import json
from pathlib import Path

import streamlit as st
import rag


# ==========================================
# 1. Page Configuration
# ==========================================

st.set_page_config(
    page_title="Egypt Gov Services Assistant",
    page_icon="🇪🇬",
    layout="wide"
)


# ==========================================
# 2. Custom Design
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #F3F7F3;
    }

    .main-title {
        color: #176B50;
        font-size: 34px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #64756C;
        text-align: center;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #176B50;
        font-size: 25px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #64756C;
        font-size: 14px;
        margin-bottom: 15px;
    }

    div[data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #DCE6DE;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    div[data-testid="stExpander"] summary {
        color: #176B50;
        font-weight: 600;
    }

    .service-item {
        color: #344B40;
        padding: 5px 8px;
        font-size: 15px;
    }

    div.stButton > button {
        background-color: #176B50;
        color: white;
        border-radius: 8px;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #12563F;
        color: white;
    }

    .status-text {
        color: #176B50;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. Header
# ==========================================

st.markdown(
    '<div class="main-title">🇪🇬 Egypt Gov Services Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Your assistant for Egyptian government services'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 4. Knowledge Base
# ==========================================

@st.cache_resource
def initialize_knowledge_base():
    return rag.build_database()


try:
    with st.spinner("Connecting to the knowledge base..."):
        collection = initialize_knowledge_base()

    st.success("The knowledge base is ready.")

except Exception as error:
    st.error(
        "Could not connect to the knowledge base. "
        "Make sure Ollama is running and the project files are available."
    )
    st.exception(error)
    st.stop()


# ==========================================
# 5. Load Available Services
# ==========================================

SERVICES_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "egypt_services_rag_organized.json"
)


@st.cache_data
def load_available_services(file_path, modified_time):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    services = data.get("services", [])

    categories = {}

    for service in services:

        category = service.get(
            "category",
            "Other Services"
        )

        service_name = service.get("service_name")

        if service_name:

            categories.setdefault(
                category,
                []
            ).append(service_name)

    return categories


# ==========================================
# 6. Available Services Section
# ==========================================

st.markdown(
    '<div class="section-title">Available Services</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Browse the government services available in our system.'
    '</div>',
    unsafe_allow_html=True
)

try:

    file_modified_time = SERVICES_FILE.stat().st_mtime

    categories = load_available_services(
        str(SERVICES_FILE),
        file_modified_time
    )

    if categories:

        for category, service_names in categories.items():

            with st.expander(
                f"📁 {category}",
                expanded=False
            ):

                for service_name in service_names:

                    st.markdown(
                        f'<div class="service-item">'
                        f'• {service_name}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    else:

        st.info("No services are currently available.")

except FileNotFoundError:

    st.error(
        "Services JSON file was not found. "
        "Please check the data folder."
    )

except Exception as error:

    st.error("Could not load available services.")
    st.exception(error)


st.divider()


# ==========================================
# 7. Chat Assistant Section
# ==========================================

st.markdown(
    '<div class="section-title">Chat Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Ask questions about Egyptian government services.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# 8. Chat History
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"**Service:** "
                        f"{source.get('service_name', 'Not available')}"
                    )

                    st.markdown(
                        f"**Category:** "
                        f"{source.get('category', 'Not available')}"
                    )

                    st.markdown(
                        f"**Verification:** "
                        f"{source.get('verification_status', 'Not available')}"
                    )

                    official_sources = source.get(
                        "official_sources",
                        ""
                    )

                    if official_sources:

                        st.markdown("**Official Sources:**")

                        for url in official_sources.split(" | "):

                            if url.strip():

                                st.markdown(
                                    f"- [{url}]({url})"
                                )

                    st.divider()


# ==========================================
# 9. User Question
# ==========================================

question = st.chat_input(
    "Ask about any government service..."
)


# ==========================================
# 10. Generate Answer
# ==========================================

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner("Searching for the answer..."):

                answer, sources = rag.ask_question(
                    collection,
                    question
                )

            st.markdown("### Answer")
            st.markdown(answer)

            if sources:

                with st.expander("Sources", expanded=True):

                    for source in sources:

                        st.markdown(
                            f"**Service:** "
                            f"{source.get('service_name', 'Not available')}"
                        )

                        st.markdown(
                            f"**Category:** "
                            f"{source.get('category', 'Not available')}"
                        )

                        st.markdown(
                            f"**Verification:** "
                            f"{source.get('verification_status', 'Not available')}"
                        )

                        official_sources = source.get(
                            "official_sources",
                            ""
                        )

                        if official_sources:

                            st.markdown("**Official Sources:**")

                            for url in official_sources.split(" | "):

                                if url.strip():

                                    st.markdown(
                                        f"- [{url}]({url})"
                                    )

                        st.divider()

            else:

                st.info("No sources were found for this question.")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as error:

            error_message = (
                "An error occurred while generating the answer. "
                "Please try again."
            )

            st.error(error_message)
            st.exception(error)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })