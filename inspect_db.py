import chromadb
import json

# 1. Connect to the existing database
# We use the same path as in your ingest.py
client = chromadb.PersistentClient(path="./db")

# 2. Get the collection
# We use the exact collection name defined in ingest.py
try:
    collection = client.get_collection(name="resume_store")
    print(f"✅ Connected to database. Found {collection.count()} resumes stored.\n")
except Exception as e:
    print(f"❌ Could not find collection. Have you run ingest.py yet? Error: {e}")
    exit()

# 3. Fetch all data (we only need metadata and documents, not embeddings)
data = collection.get(include=["metadatas", "documents"])

# 4. Loop through and nicely print the details
print("="*60)
print(f"{'FILENAME':<30} | {'CANDIDATE NAME':<20} | {'EXP':<5} | {'SKILLS (Preview)'}")
print("="*60)

# We use a set to track processed files so we don't print every single chunk
# (Since one PDF is split into many chunks, we only want to see the metadata once per file)
seen_files = set()

for i, meta in enumerate(data['metadatas']):
    source_file = meta.get('source', 'Unknown')
    
    if source_file not in seen_files:
        name = meta.get('name', 'Unknown')
        email = meta.get('email', 'Unknown')
        exp = meta.get('years_exp', 0)
        skills = meta.get('skills', '[]')
        
        # Truncate skills for display
        if len(skills) > 50:
            skills_preview = skills[:50] + "..."
        else:
            skills_preview = skills

        print(f"{source_file[:30]:<30} | {name[:20]:<20} | {str(exp):<5} | {skills_preview}")
        seen_files.add(source_file)

print("="*60)
print("\n🔍 Detailed Inspection of the LAST processed resume:")
if data['metadatas']:
    last_meta = data['metadatas'][-1]
    print(json.dumps(last_meta, indent=2))