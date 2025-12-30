
# import os
# from typing import Dict, Any, List
# from dotenv import load_dotenv

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser

# load_dotenv()

# # Verify API Key
# if not os.getenv("GROQ_API_KEY"):
#     raise ValueError("GROQ_API_KEY not found in environment variables.")

# DB_PATH = "./db"
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# class ResumeRAG:
#     def __init__(self):
#         self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#         self.vectorstore = Chroma(
#             persist_directory=DB_PATH, 
#             embedding_function=self.embeddings,
#             collection_name="resume_store"
#         )
#         self.llm = ChatGroq(
#             model="llama-3.3-70b-versatile",
#             temperature=0.0
#         )
        
#     def classify_intent(self, query: str) -> Dict[str, Any]:
#         """
#         Determines if the query is for a specific person, a filtered search, or general synthesis.
#         """
#         parser = JsonOutputParser()
        
#         # Note: We use double braces {{ }} to escape JSON structure in PromptTemplate
#         prompt = PromptTemplate(
#             template="""
#             You are an AI router for a Resume Database. User Query: "{query}"
            
#             Determine the intent and extract filters. Return ONLY a JSON object.
            
#             Rules:
#             1. If the user asks about a specific person (e.g., "Who is John?", "Tell me about Sarah"),
#                set "type": "person", "filter": {{ "name": "Exact Name From Query" }}.
#             2. If the user filters by attributes (e.g., "Python devs with 5+ years exp"),
#                set "type": "filter". Construct a Mongo-style selector for metadata.
#                Available metadata fields: "years_exp" (int), "skills" (string contains).
#                Example: {{ "years_exp": {{ "$gte": 5 }} }}.
#                Note: For skills, return empty filter unless specific.
#             3. If the query is general (e.g., "Summarize all java devs", "Find me the best candidate"),
#                set "type": "general", "filter": {{}}.
            
#             JSON Output:
#             """,
#             input_variables=["query"],
#         )
        
#         chain = prompt | self.llm | parser
#         try:
#             return chain.invoke({"query": query})
#         except Exception as e:
#             # Fallback
#             print(f"Intent classification failed: {e}")
#             return {"type": "general", "filter": {}}

#     def retrieve_documents(self, query: str, intent: Dict[str, Any]) -> List[Any]:
#         k = 10 if intent['type'] == 'general' else 5
        
#         filter_dict = intent.get('filter', {})
        
#         # Clean up empty filters
#         if not filter_dict:
#             filter_dict = None
            
#         # Chroma/LangChain integration sometimes struggles with complex $gte filters passed directly.
#         # Ideally, we pass them as-is. If errors occur, we fallback to no filter.
#         try:
#             results = self.vectorstore.similarity_search(
#                 query,
#                 k=k,
#                 filter=filter_dict
#             )
#             return results
#         except Exception as e:
#             print(f"Complex filter failed ({e}), retrying without filter...")
#             return self.vectorstore.similarity_search(query, k=k)

#     def answer_query(self, query: str):
#         # 1. Intent Classification
#         intent = self.classify_intent(query)
        
#         # 2. Retrieval
#         docs = self.retrieve_documents(query, intent)

#         if not docs:
#             return "No matching resumes found based on your criteria.", intent, []

#         # 3. Generation
#         context_text = "\n\n".join([
#             f"Source: {doc.metadata.get('source', 'Unknown')}\n"
#             f"Name: {doc.metadata.get('name', 'Unknown')}\n"
#             f"Experience: {doc.metadata.get('years_exp', 0)} years\n"
#             f"Content: {doc.page_content}" 
#             for doc in docs
#         ])
        
#         system_prompt = f"""
#         You are an expert HR Assistant. Answer the user's question based ONLY on the context below.
        
#         Context:
#         {context_text}
        
#         User Question: {query}
        
#         Answer professionally. Cite the candidates' names when mentioning specific details.
#         """
        
#         response = self.llm.invoke(system_prompt)
#         return response.content, intent, docs


###################################################################


import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

load_dotenv()

# Verify API Key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

DB_PATH = "./db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class ResumeRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=self.embeddings,
            collection_name="resume_store"
        )
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0
        )
        
    def _get_all_metadata(self) -> List[Dict]:
        """Helper to fetch the registry of all candidates for filtering."""
        try:
            # We fetch IDs to map them, but primarily we need metadatas
            data = self.vectorstore.get(include=['metadatas'])
            unique_candidates = {}
            
            for meta in data['metadatas']:
                source = meta.get('source')
                if source and source not in unique_candidates:
                    unique_candidates[source] = meta
            
            return list(unique_candidates.values())
        except Exception as e:
            print(f"Error fetching metadata: {e}")
            return []

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Decides if we need to:
        1. LIST/FILTER (Look at everyone's metadata)
        2. COMPARE/SPECIFIC (Look at full text of specific people)
        3. GENERAL (Standard RAG search)
        """
        parser = JsonOutputParser()
        
        prompt = PromptTemplate(
            template="""
            Analyze the User Query: "{query}"
            
            Classify into one of these types and return JSON:
            
            1. "type": "list_filter"
               - Use this when user asks to "find", "list", "show me", "count" candidates with specific skills/experience.
               - Extract "keywords" (e.g., ["Python", "Java", "5 years"]).
               
            2. "type": "compare_specific"
               - Use this when user asks about SPECIFIC people by name (e.g., "Compare Harish and Amit", "Tell me about Neha").
               - Extract "target_names" (list of names mentioned).
               
            3. "type": "general"
               - Use for conceptual questions (e.g., "What is a good resume?", "Summarize the database").
               
            JSON Output:
            """,
            input_variables=["query"],
        )
        
        try:
            chain = prompt | self.llm | parser
            return chain.invoke({"query": query})
        except Exception:
            return {"type": "general"}

    def handle_list_query(self, query: str, intent: Dict) -> tuple:
        """Strategy: Fetch ALL metadata, filter loosely in Python, let LLM refine."""
        all_candidates = self._get_all_metadata()
        
        # 1. Pre-filtering in Python to reduce token usage (optional but recommended for 200+ files)
        # We'll construct a text representation of each candidate for the LLM
        candidate_summaries = []
        keywords = [k.lower() for k in intent.get('keywords', [])]
        
        for cand in all_candidates:
            # Simple keyword check (if keywords exist)
            cand_str = (
                f"Name: {cand.get('name', 'Unknown')}, "
                f"Exp: {cand.get('years_exp', 0)}, "
                f"Skills: {cand.get('skills', '')}, "
                f"File: {cand.get('source', '')}"
            ).lower()
            
            # If no keywords, include everyone. If keywords, require at least one match.
            if not keywords or any(k in cand_str for k in keywords):
                candidate_summaries.append(
                    f"- Name: {cand.get('name', 'Unknown')}\n"
                    f"  Experience: {cand.get('years_exp', 0)} years\n"
                    f"  Skills: {cand.get('skills', 'N/A')}\n"
                    f"  Source File: {cand.get('source')}"
                )

        # 2. Limit to avoid token overflow (Top 30 most relevant if too many)
        context_text = "\n\n".join(candidate_summaries[:30])
        
        system_prompt = f"""
        You are an HR Analyst. The user wants a list of candidates.
        Below is the raw data of candidates that matched the rough search.
        
        Your Job:
        1. Filter this list strictly based on the user's request: "{query}"
        2. Present the final list clearly with Name, Years of Exp, and matching Skills.
        3. If no one strictly matches, say so.
        
        Candidate Data:
        {context_text}
        """
        
        response = self.llm.invoke(system_prompt)
        return response.content, candidate_summaries[:10] # Return top 10 as "source docs"

    def handle_compare_query(self, query: str, intent: Dict) -> tuple:
        """Strategy: Find specific documents for the named people and compare them."""
        target_names = [n.lower() for n in intent.get('target_names', [])]
        all_candidates = self._get_all_metadata()
        
        # 1. Find the filenames matching the requested names
        matched_files = []
        for cand in all_candidates:
            c_name = cand.get('name', '').lower()
            # fuzzy check: if "harish" is in "harish jangid"
            if any(t in c_name for t in target_names):
                matched_files.append(cand.get('source'))
        
        if not matched_files:
            return "I couldn't find resumes matching those names. Please check the spelling or view the full list.", []

        # 2. Retrieve FULL text for these specific files
        # We assume 1 resume = ~5 chunks. We fetch them using metadata filter.
        docs = self.vectorstore.get(where={"source": {"$in": matched_files}})
        
        # Reconstruct full text from chunks
        full_profiles = {}
        for meta, content in zip(docs['metadatas'], docs['documents']):
            name = meta.get('name', 'Unknown')
            if name not in full_profiles:
                full_profiles[name] = ""
            full_profiles[name] += content + "\n"

        context_text = ""
        for name, text in full_profiles.items():
            context_text += f"=== PROFILE: {name} ===\n{text[:3000]}\n\n" # Limit text per person

        system_prompt = f"""
        You are an HR Assistant. Compare the following candidates based on the user's request.
        
        User Request: "{query}"
        
        Profiles:
        {context_text}
        
        Provide a detailed side-by-side comparison (or bullet points) of their strengths, weaknesses, and suitability.
        """
        
        response = self.llm.invoke(system_prompt)
        
        # Create dummy docs for UI display
        source_docs = [Document(page_content="Full Profile Used", metadata={"source": f}) for f in matched_files]
        return response.content, source_docs

    def handle_general_query(self, query: str) -> tuple:
        """Strategy: Standard Semantic Search (good for 'What does candidate X do?')"""
        docs = self.vectorstore.similarity_search(query, k=5)
        
        context_text = "\n\n".join([
            f"Source: {doc.metadata.get('source')}\nContent: {doc.page_content}" 
            for doc in docs
        ])
        
        system_prompt = f"""
        Answer the question based on the resume snippets below.
        
        Context:
        {context_text}
        
        Question: {query}
        """
        response = self.llm.invoke(system_prompt)
        return response.content, docs

    def answer_query(self, query: str):
        # 1. Router
        intent = self.classify_intent(query)
        intent_type = intent.get('type', 'general')
        
        print(f"DEBUG: Intent Detected -> {intent_type}") # Helpful for debugging
        
        # 2. Dispatch to Strategy
        if intent_type == 'list_filter':
            response, docs = self.handle_list_query(query, intent)
        elif intent_type == 'compare_specific':
            response, docs = self.handle_compare_query(query, intent)
        else:
            response, docs = self.handle_general_query(query)
            
        return response, intent, docs