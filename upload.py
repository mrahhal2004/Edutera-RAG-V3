import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 1. Setup Environment
load_dotenv()
chroma_client = chromadb.PersistentClient(path="./my_chroma_db") 

# Clear old collection if exists (Fresh Start)
try:
    chroma_client.delete_collection("unit1_math_content")
except:
    pass

# Create new collection
collection = chroma_client.create_collection(name="unit1_math_content")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Smart Parser (Looks for $$$$)
def parse_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chunks = []
    
    # State Counters
    current_unit = 1
    current_lesson = 0
    current_skill_id = 0 
    current_skill_name = "Introduction"
    current_text = []
    
    for line in lines:
        line = line.strip()
        
        # A. Detect Start of a Lesson (# Header)
        if line.startswith("# "):
            # Save previous chunk if exists
            if current_text:
                chunks.append({
                    "text": "\n".join(current_text),
                    "metadata": {
                        "unit_id": current_unit,
                        "lesson_id": current_lesson,
                        "skill_id": current_skill_id,
                        "skill_name": current_skill_name,
                        "chunk_type": "content"
                    }
                })
                current_text = []
            
            # Update Lesson State
            current_lesson += 1
            current_skill_id += 1 
            # Remove # and clean whitespace
            current_skill_name = line.replace("#", "").strip()
            # Start fresh text for new lesson introduction
            current_text.append(line)
            continue

        # B. Detect MAIN SKILL ($$$$ Header)
        if line.startswith("$$$$"):
            # Save previous chunk (Previous skill is done)
            if current_text:
                chunks.append({
                    "text": "\n".join(current_text),
                    "metadata": {
                        "unit_id": current_unit,
                        "lesson_id": current_lesson,
                        "skill_id": current_skill_id,
                        "skill_name": current_skill_name,
                        "chunk_type": "content"
                    }
                })
                current_text = []
            
            # Update Skill State
            current_skill_id += 1
            # Remove $$$$ markers
            current_skill_name = line.replace("$$$$", "").strip()
            # Add title to text context as well
            current_text.append(line.replace("$$$$", "## ")) 
            continue
            
        # C. Everything else (###, text, bullets) -> Just append!
        if line:
            current_text.append(line)
            
    # Save the very last chunk
    if current_text:
        chunks.append({
            "text": "\n".join(current_text),
            "metadata": {
                "unit_id": current_unit,
                "lesson_id": current_lesson,
                "skill_id": current_skill_id,
                "skill_name": current_skill_name,
                "chunk_type": "content"
            }
        })
        
    return chunks

# 3. Main Execution
def upload_data():
    print("📂 Parsing Cleaned Markdown (Looking for $$$$)...")
    # Make sure to read the NEW file
    chunks = parse_markdown("unit1_clean.md")
    
    print(f"🧩 Found {len(chunks)} Highly Focused Skills.")
    
    ids = []
    documents = []
    metadatas = []
    
    print("🧠 Generating Embeddings...")
    for i, chunk in enumerate(chunks):
        ids.append(f"cid_{i}")
        documents.append(chunk['text'])
        metadatas.append(chunk['metadata'])

    # Batch Process
    batch_size = 10
    total_batches = len(documents) // batch_size + 1
    
    print("💾 Saving to ChromaDB...")
    for i in range(total_batches):
        start = i * batch_size
        end = start + batch_size
        if start >= len(documents): break
        
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embedding_model.encode(documents[start:end]).tolist(),
            metadatas=metadatas[start:end]
        )
        print(f"   - Batch {i+1}/{total_batches} saved.")

    print("✅ Clean Data Uploaded Successfully!")

if __name__ == "__main__":
    upload_data()