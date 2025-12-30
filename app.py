# import streamlit as st
# import os
# import shutil
# from ingest import run_ingestion
# from rag_engine import ResumeRAG

# # Page Config
# st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")
# st.title("📄 AI Resume Analyst")

# # Initialize Session State
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "rag_engine" not in st.session_state:
#     st.session_state.rag_engine = ResumeRAG()

# # Sidebar: Ingestion Control
# with st.sidebar:
#     st.header("Data Management")
    
#     uploaded_files = st.file_uploader(
#         "Upload Resumes (PDF)", 
#         type="pdf", 
#         accept_multiple_files=True
#     )
    
#     if uploaded_files:
#         upload_dir = "./resumes"
#         if not os.path.exists(upload_dir):
#             os.makedirs(upload_dir)
            
#         progress_text = "Saving files..."
#         my_bar = st.progress(0, text=progress_text)
        
#         for i, file in enumerate(uploaded_files):
#             with open(os.path.join(upload_dir, file.name), "wb") as f:
#                 f.write(file.getbuffer())
#             my_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
            
#         st.success(f"Uploaded {len(uploaded_files)} files!")
    
#     if st.button("🚀 Process/Ingest Resumes"):
#         with st.spinner("Ingesting resumes... Check terminal for detailed logs."):
#             run_ingestion()
#         st.success("Ingestion Complete! The brain has been updated.")
#         # Reload engine to pick up new data
#         st.session_state.rag_engine = ResumeRAG()

# # Main Chat Interface
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if "reasoning" in message:
#             with st.expander("🔍 Reasoning & Sources"):
#                 st.json(message["reasoning"]["intent"])
#                 st.markdown("**Sources Used:**")
#                 for src in message["reasoning"]["sources"]:
#                     st.markdown(f"- `{src}`")

# if prompt := st.chat_input("Ask about candidates (e.g., 'Find Python devs with 5+ years exp')"):
#     # User Message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # AI Response
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response_text, intent, docs = st.session_state.rag_engine.answer_query(prompt)
            
#             # Extract unique sources
#             sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))
            
#             st.markdown(response_text)
            
#             # Reasoning Data
#             reasoning_data = {
#                 "intent": intent,
#                 "sources": sources
#             }
            
#             with st.expander("🔍 Reasoning & Sources"):
#                 st.json(intent)
#                 st.markdown("**Sources Used:**")
#                 for src in sources:
#                     st.markdown(f"- `{src}`")

#     # Append to history
#     st.session_state.messages.append({
#         "role": "assistant", 
#         "content": response_text,
#         "reasoning": reasoning_data
#     })
#################################################################################
# import streamlit as st
# import os
# from ingest import run_ingestion
# from rag_engine import ResumeRAG

# # Page Config
# st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")
# st.title("📄 AI Resume Analyst")

# # Initialize Session State
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "rag_engine" not in st.session_state:
#     # Check if DB exists before initializing to prevent errors
#     if os.path.exists("./db"):
#         st.session_state.rag_engine = ResumeRAG()
#     else:
#         st.session_state.rag_engine = None

# # Sidebar: Ingestion Control
# with st.sidebar:
#     st.header("Data Management")
    
#     uploaded_files = st.file_uploader(
#         "Upload Resumes (PDF)", 
#         type="pdf", 
#         accept_multiple_files=True
#     )
    
#     if uploaded_files:
#         upload_dir = "./resumes"
#         if not os.path.exists(upload_dir):
#             os.makedirs(upload_dir)
            
#         progress_text = "Saving files..."
#         my_bar = st.progress(0, text=progress_text)
        
#         for i, file in enumerate(uploaded_files):
#             with open(os.path.join(upload_dir, file.name), "wb") as f:
#                 f.write(file.getbuffer())
#             my_bar.progress((i + 1) / len(uploaded_files), text=progress_text)
            
#         st.success(f"Uploaded {len(uploaded_files)} files!")
    
#     if st.button("🚀 Process/Ingest Resumes"):
#         with st.spinner("Ingesting resumes... Check terminal for detailed logs."):
#             run_ingestion()
#         st.success("Ingestion Complete! The brain has been updated.")
#         # Reload engine to pick up new data
#         st.session_state.rag_engine = ResumeRAG()

# # Main Chat Interface
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#         if "reasoning" in message:
#             with st.expander("🔍 Reasoning & Sources"):
#                 st.json(message["reasoning"]["intent"])
#                 st.markdown("**Sources Used:**")
#                 for src in message["reasoning"]["sources"]:
#                     st.markdown(f"- `{src}`")

# if prompt := st.chat_input("Ask about candidates (e.g., 'Find Python devs with 5+ years exp')"):
#     # User Message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # AI Response
#     with st.chat_message("assistant"):
#         if st.session_state.rag_engine is None:
#             st.error("Please ingest some resumes first!")
#         else:
#             with st.spinner("Thinking..."):
#                 response_text, intent, docs = st.session_state.rag_engine.answer_query(prompt)
                
#                 # Extract unique sources
#                 sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))
                
#                 st.markdown(response_text)
                
#                 # Reasoning Data
#                 reasoning_data = {
#                     "intent": intent,
#                     "sources": sources
#                 }
                
#                 with st.expander("🔍 Reasoning & Sources"):
#                     st.json(intent)
#                     st.markdown("**Sources Used:**")
#                     for src in sources:
#                         st.markdown(f"- `{src}`")

#     # Append to history
#     if st.session_state.rag_engine is not None:
#         st.session_state.messages.append({
#             "role": "assistant", 
#             "content": response_text,
#             "reasoning": reasoning_data
#         })


##############################################################

import streamlit as st
import os
from ingest import run_ingestion
from rag_engine import ResumeRAG

st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")
st.title("📄 Expert AI Resume Analyst")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    if os.path.exists("./db"):
        st.session_state.rag_engine = ResumeRAG()
    else:
        st.session_state.rag_engine = None

# Sidebar
with st.sidebar:
    st.header("Data Management")
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        upload_dir = "./resumes"
        if not os.path.exists(upload_dir): os.makedirs(upload_dir)
        for file in uploaded_files:
            with open(os.path.join(upload_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Uploaded {len(uploaded_files)} files!")
    
    if st.button("🚀 Process/Ingest Resumes"):
        with st.spinner("Processing... This may take a while for large batches."):
            run_ingestion()
        st.success("Ingestion Complete!")
        st.session_state.rag_engine = ResumeRAG()

# Chat Area
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "reasoning" in message:
            with st.expander("🔍 Analysis Details"):
                st.write(f"**Intent:** {message['reasoning'].get('intent', {}).get('type', 'Unknown')}")
                st.write("**Sources/Context:**")
                for src in message["reasoning"]["sources"]:
                    st.text(src)

if prompt := st.chat_input("Ask: 'List Python devs', 'Compare Harish and Amit', 'Who has 10+ years exp?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.rag_engine is None:
            st.error("⚠️ No database found. Please upload resumes and click Process!")
        else:
            with st.spinner("Analyzing candidate database..."):
                response_text, intent, docs = st.session_state.rag_engine.answer_query(prompt)
                
                # Helper to extract source names safely
                source_names = []
                for doc in docs:
                    if isinstance(doc, str): # If we passed raw strings in List mode
                        source_names.append(doc.split('\n')[0]) # Just grab the first line (Name)
                    else: # If it's a Document object
                        source_names.append(doc.metadata.get('source', 'Unknown'))
                
                # Remove duplicates
                source_names = list(set(source_names))
                
                st.markdown(response_text)
                
                reasoning_data = {
                    "intent": intent,
                    "sources": source_names
                }
                
                with st.expander("🔍 Analysis Details"):
                    st.write(f"**Intent Detected:** `{intent.get('type')}`")
                    st.write("**Sources Identified:**")
                    for src in source_names:
                        st.text(f"- {src}")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "reasoning": reasoning_data
    })