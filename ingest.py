# import os
# import json
# import logging
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import List, Dict, Any

# from dotenv import load_dotenv
# # Switched to PyMuPDFLoader for better symbol handling
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# # Switched to Groq for metadata extraction
# from langchain_groq import ChatGroq
# from langchain_core.documents import Document

# # Load Environment
# load_dotenv()

# # Verify API Keys
# if not os.getenv("GROQ_API_KEY"):
#     raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")

# # Configuration
# DB_PATH = "./db"
# RESUME_DIR = "./resumes"
# MAX_WORKERS = 5
# BATCH_SIZE = 50
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# # Logging Setup
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Initialize Embeddings
# embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# # Initialize Groq for Metadata Extraction
# # We use a fast model for extraction
# metadata_llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0.0,
#     max_retries=3
# )

# def get_existing_files(vectorstore: Chroma) -> set:
#     """Query DB to find which files are already indexed."""
#     try:
#         data = vectorstore.get(include=['metadatas'])
#         existing_sources = set()
#         for meta in data['metadatas']:
#             if meta and 'source' in meta:
#                 existing_sources.add(os.path.basename(meta['source']))
#         return existing_sources
#     except Exception:
#         return set()

# def extract_metadata(text: str) -> Dict[str, Any]:
#     """Uses Groq (Llama 3) to extract structured metadata from resume text."""
#     prompt = f"""
#     You are a precise data extraction AI. Analyze the resume text below and extract details into a raw JSON object.
    
#     Rules:
#     1. Output ONLY valid JSON. No markdown formatting (no ```json).
#     2. Do not include introductory text.
    
#     Fields required:
#     - "name": (string) Candidate's full name. If unknown, use "Unknown".
#     - "email": (string) Email address. Use "Unknown" if not found.
#     - "skills": (list of strings) Key technical skills.
#     - "years_exp": (int) Total years of professional experience. Return 0 if not found.
#     - "summary": (string) A brief 2-sentence professional summary.

#     Resume Text:
#     {text[:4000]} 
#     """
    
#     try:
#         response = metadata_llm.invoke(prompt)
#         content = response.content.strip()
        
#         # Clean potential markdown or extra text
#         if "```json" in content:
#             content = content.split("```json")[1].split("```")[0].strip()
#         elif "```" in content:
#             content = content.split("```")[1].split("```")[0].strip()
            
#         # Parse JSON
#         return json.loads(content)
#     except Exception as e:
#         logger.error(f"Metadata extraction failed: {e}")
#         return {
#             "name": "Unknown", 
#             "email": "Unknown", 
#             "skills": [], 
#             "years_exp": 0, 
#             "summary": "Metadata extraction failed"
#         }

# def process_single_resume(file_path: str) -> List[Document]:
#     """Reads PDF using PyMuPDF, extracts metadata, and chunks text."""
#     try:
#         filename = os.path.basename(file_path)
        
#         # Updated to PyMuPDFLoader
#         loader = PyMuPDFLoader(file_path)
#         pages = loader.load()
#         full_text = " ".join([p.page_content for p in pages])
        
#         if not full_text.strip():
#             logger.error(f"Empty text in file: {filename}")
#             return []

#         # Extract Metadata
#         meta = extract_metadata(full_text)
        
#         # Add source filename to metadata
#         meta['source'] = filename
        
#         # Flatten skills list to string for Chroma filtering compatibility
#         if isinstance(meta.get('skills'), list):
#             meta['skills'] = ", ".join(meta['skills']) 

#         # Chunking
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = splitter.split_text(full_text)
        
#         docs = []
#         for chunk in chunks:
#             docs.append(Document(page_content=chunk, metadata=meta))
            
#         return docs

#     except Exception as e:
#         logger.error(f"Failed to process {file_path}: {e}")
#         return []

# def run_ingestion():
#     if not os.path.exists(DB_PATH):
#         os.makedirs(DB_PATH)
    
#     # Initialize Persistent DB
#     vectorstore = Chroma(
#         persist_directory=DB_PATH, 
#         embedding_function=embeddings,
#         collection_name="resume_store"
#     )
    
#     existing_files = get_existing_files(vectorstore)
#     logger.info(f"Found {len(existing_files)} existing files in DB.")
    
#     all_files = [f for f in os.listdir(RESUME_DIR) if f.lower().endswith('.pdf')]
#     files_to_process = [os.path.join(RESUME_DIR, f) for f in all_files if f not in existing_files]
    
#     if not files_to_process:
#         logger.info("No new files to process.")
#         return

#     logger.info(f"Processing {len(files_to_process)} new resumes using Groq...")
    
#     docs_buffer = []
#     total_processed = 0
    
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         future_to_file = {executor.submit(process_single_resume, f): f for f in files_to_process}
        
#         for future in as_completed(future_to_file):
#             docs = future.result()
#             if docs:
#                 docs_buffer.extend(docs)
            
#             # Batch Saving Logic
#             if len(docs_buffer) >= BATCH_SIZE * 5:
#                 logger.info(f"Committing batch of {len(docs_buffer)} chunks...")
#                 vectorstore.add_documents(docs_buffer)
#                 docs_buffer = [] 
                
#             total_processed += 1
#             if total_processed % 10 == 0:
#                 logger.info(f"Progress: {total_processed}/{len(files_to_process)} resumes processed.")

#     # Final Commit
#     if docs_buffer:
#         logger.info(f"Committing final batch of {len(docs_buffer)} chunks...")
#         vectorstore.add_documents(docs_buffer)
        
#     logger.info("Ingestion Complete.")

# if __name__ == "__main__":
#     if not os.path.exists(RESUME_DIR):
#         os.makedirs(RESUME_DIR)
#         print(f"Created {RESUME_DIR}. Please add PDF files there.")
#     else:
#         run_ingestion()

#####################################################################

import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# Load Environment
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

# Configuration
DB_PATH = "./db"
RESUME_DIR = "./resumes"
MAX_WORKERS = 2  # Keep low to avoid rate limits
BATCH_SIZE = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
metadata_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, max_retries=3)

def get_existing_files(vectorstore: Chroma) -> set:
    try:
        data = vectorstore.get(include=['metadatas'])
        existing = set()
        for meta in data['metadatas']:
            if meta and 'source' in meta:
                existing.add(os.path.basename(meta['source']))
        return existing
    except Exception:
        return set()

def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extracts structured data. 
    CRITICAL: We enforce lowercase skills for easier searching later.
    """
    prompt = f"""
    You are a Resume Parser. Extract data into strict JSON format.
    
    Resume Text (first 4000 chars):
    {text[:4000]}
    
    Required JSON Fields:
    - "name": (string) "Unknown" if not found.
    - "email": (string) "Unknown" if not found.
    - "years_exp": (int) 0 if not found.
    - "skills": (list of strings) e.g., ["python", "java", "sql"]. convert to lowercase.
    - "seniority": (string) "Junior" (0-2 yrs), "Mid" (3-5 yrs), "Senior" (5+ yrs).
    - "summary": (string) Short 2-sentence summary.
    
    Output ONLY JSON.
    """
    
    try:
        time.sleep(1.5) # Rate limit safety
        response = metadata_llm.invoke(prompt)
        content = response.content.strip()
        
        # Cleaner JSON parsing
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(content)
        
        # Data Normalization (Crucial for the "List" feature)
        if 'skills' in data and isinstance(data['skills'], list):
            data['skills'] = [s.lower().strip() for s in data['skills']]
            # Convert list to string for ChromaDB storage (Chroma basic doesn't support list types in metadata well)
            # We store as comma-separated string for storage, but the LLM created the list first to ensure quality.
            data['skills'] = ", ".join(data['skills'])
            
        return data
    except Exception as e:
        logger.error(f"Metadata Error: {e}")
        return {
            "name": "Unknown", "email": "Unknown", "years_exp": 0, 
            "skills": "", "seniority": "Unknown", "summary": "Failed to extract"
        }

def process_single_resume(file_path: str) -> List[Document]:
    try:
        filename = os.path.basename(file_path)
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        full_text = " ".join([p.page_content for p in pages])
        
        if not full_text.strip():
            logger.warning(f"Empty text: {filename}")
            return []

        meta = extract_metadata(full_text)
        meta['source'] = filename
        
        # Recursive splitting is best for resumes to keep sections together
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(full_text)
        
        docs = []
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata=meta))
            
        return docs

    except Exception as e:
        logger.error(f"File Error {file_path}: {e}")
        return []

def run_ingestion():
    # Only creating DB folder if needed, Chroma handles the rest
    if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)
    
    vectorstore = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings,
        collection_name="resume_store"
    )
    
    existing = get_existing_files(vectorstore)
    logger.info(f"Existing files: {len(existing)}")
    
    all_files = [f for f in os.listdir(RESUME_DIR) if f.lower().endswith('.pdf')]
    to_process = [os.path.join(RESUME_DIR, f) for f in all_files if f not in existing]
    
    if not to_process:
        logger.info("All caught up! No new files.")
        return

    logger.info(f"Ingesting {len(to_process)} new resumes...")
    
    docs_buffer = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(process_single_resume, f): f for f in to_process}
        
        for i, future in enumerate(as_completed(future_map)):
            docs = future.result()
            if docs:
                docs_buffer.extend(docs)
                
            # Batch commit every 50 chunks
            if len(docs_buffer) >= BATCH_SIZE:
                vectorstore.add_documents(docs_buffer)
                docs_buffer = []
                logger.info(f"Saved batch. Progress: {i+1}/{len(to_process)}")

    if docs_buffer:
        vectorstore.add_documents(docs_buffer)
    
    logger.info("✅ Ingestion Done!")

if __name__ == "__main__":
    if not os.path.exists(RESUME_DIR):
        os.makedirs(RESUME_DIR)
        print(f"Created {RESUME_DIR}. Add PDFs here.")
    else:
        run_ingestion()