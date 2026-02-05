import uvicorn
import hashlib
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

# Import your existing custom logic
from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from utils.logging import logger

# 1. Initialize FastAPI App
app = FastAPI(
    title="ChattyDoc API",
    description="Backend API for multi-agent document research and verification",
    version="1.0.0"
)

# 2. Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize Core Components (Singletons)
processor = DocumentProcessor()
retriever_builder = RetrieverBuilder()
workflow = AgentWorkflow()

# Global variables to simulate session state
GLOBAL_STATE = {
    "retriever": None,
    "file_hashes": frozenset()
}

# --- Helper Function for Hashing ---
async def _get_api_file_hashes(files: List[UploadFile]) -> frozenset:
    hashes = set()
    for file in files:
        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()
        hashes.add(file_hash)
        await file.seek(0)  # Reset pointer so processor can read it
    return frozenset(hashes)

# 4. API Endpoints

@app.get("/")
async def root():
    return {"message": "ChattyDoc API is online. Go to /docs for Swagger UI."}

@app.post("/chat")
async def chat_endpoint(
    question: str = Form(...),
    files: Optional[List[UploadFile]] = File(None)
):
    global GLOBAL_STATE

    try:
        # Step A: Document Processing & Indexing
        if files and len(files) > 0:
            current_hashes = await _get_api_file_hashes(files)

            # Only rebuild if files changed or no retriever exists
            if GLOBAL_STATE["retriever"] is None or current_hashes != GLOBAL_STATE["file_hashes"]:
                logger.info("Building new retriever for API request...")
                
                # --- Compatibility Layer: Convert FastAPI files to 'Streamlit-like' objects ---
                sync_files = []
                for f in files:
                    content = await f.read()
                    # Wrap bytes in a buffer that supports the Buffer API
                    buf = io.BytesIO(content)
                    # Add attributes your DocumentProcessor expects (.name and .size)
                    buf.name = f.filename
                    buf.size = len(content) 
                    sync_files.append(buf)
                
                # Process the "faked" sync files
                chunks = processor.process(sync_files)
                
                if not chunks:
                    raise HTTPException(status_code=400, detail="No text content extracted.")

                # Build hybrid retriever
                new_retriever = retriever_builder.build_hybrid_retriever(chunks)
                
                # Update State
                GLOBAL_STATE["retriever"] = new_retriever
                GLOBAL_STATE["file_hashes"] = current_hashes
            else:
                logger.info("Using cached retriever.")

        # Step B: Safety Check
        retriever = GLOBAL_STATE["retriever"]
        if retriever is None:
            raise HTTPException(
                status_code=400, 
                detail="No documents found. Please upload documents."
            )

        # Step C: Execution
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        logger.info(f"Executing workflow for question: {question}")
        
        # Run your LangGraph / Agentic pipeline
        result = workflow.full_pipeline(
            question=question,
            retriever=retriever
        )

        # Step D: Response
        return {
            "status": "success",
            "data": {
                "answer": result.get("draft_answer"),
                "verification": result.get("verification_report"),
                "metadata": {
                    "files_indexed": len(GLOBAL_STATE["file_hashes"]),
                    "sources_consulted": len(result.get("source_documents", []))
                }
            }
        }

    except Exception as e:
        logger.error(f"Critical API Error: {str(e)}")
        # Return 500 for backend logic crashes, 400 for user input issues
        raise HTTPException(status_code=500, detail=str(e))

# 5. Execution Block
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)