from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from index_dataset import load_existing_index
from Rag_engine import ask_gem_guide, get_artifact_metadata
from load_model import get_llm
import time
import psutil
from utils import get_gpu_stats
from fastapi.responses import StreamingResponse
from langchain_core.messages import SystemMessage, HumanMessage
from Rag_engine import build_rag_prompt, retrieve_by_artifact_id, detect_language

resources = {}

@asynccontextmanager
async def startup(app: FastAPI):
    index, chroma_collection = load_existing_index()
    resources["index"] = index
    resources["chroma"] = chroma_collection
    # resources["llm"] = get_llm()
    # Pre-warm the model — load it into VRAM now
    print("Pre-warming LLM...")
    llm = get_llm("batiai/gemma4-e2b:q4", 670)
    llm.invoke([
        SystemMessage(content="Hello"),
        HumanMessage(content="Hi")
    ])
    # Also warm the text-mode config
    llm2 = get_llm("batiai/gemma4-e2b:q4", 800)
    llm2.invoke([
        SystemMessage(content="Hello"),
        HumanMessage(content="Hi")
    ])
    print("LLM pre-warmed and ready!")
    yield

app = FastAPI(title="GEM Tour RAG API", lifespan=startup)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question:str
    artifact_id: str | None

@app.post("/ask")
def ask_guide(request: QueryRequest):
    try:
        start = time.time()
        result = ask_gem_guide(
            resources["index"],
            resources["chroma"],
            # resources["llm"],
            "batiai/gemma4-e2b:q4",
            request.question,
            request.artifact_id,
            verbose=False
        )
        elapsed = time.time() - start
        return {'result': result['raw_response'], 'time': round(elapsed,2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/artifact/{artifact_id}")
def get_artifact(artifact_id: str):
    metadata = get_artifact_metadata(resources["chroma"], artifact_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return metadata

###################################################################

MODELS = {
    # "gemma3n-e4b": "i82blikeu/gemma-3n-E4B-it-GGUF:Q3_K_M",
    "gemma4-e2b-q4": "batiai/gemma4-e2b:q4",
    # "qwen3-4b": "qwen3:4b",
    # "qwen3.5-4b": "qwen3.5:4b",
    # "llama3.2-3b": "llama3.2:3b",
}

test_questions = [
    # Camera mode - English
    {
        "question": "What can you tell me about this golden shrine?",
        "artifact_id": "Little Golden Shrine"
    },
    # Camera mode - Arabic
    {
        "question": "ما قصة هذا التمثال ولأي فترة تاريخية ينتمي؟",
        "artifact_id": "Ramesses III as Standard Bearer"
    },
    # Camera mode - English
    {
        "question": "Who made this statue and why was it important?",
        "artifact_id": "Scribe Statue of Paramessu"
    },
    # Camera mode - Arabic
    {
        "question": "من هي صاحبة هذا التمثال وما علاقتها بالملك؟",
        "artifact_id": "Queen Mutnofret Statue"
    },
    # Camera mode - English
    {
        "question": "What god does this statue represent and what is its significance?",
        "artifact_id": "Re-Horakhty Statue"
    },
    # Camera mode - Arabic
    {
        "question": "أخبرني عن هذا التمثال المزدوج وما الذي يحمله؟",
        "artifact_id": "Double Statue of Ramesses II"
    },
    # Text mode - English
    {
        "question": "What artifacts related to Ramesses II are in the museum?",
        "artifact_id": None
    },
    # Text mode - Arabic
    {
        "question": "ما هي التماثيل المصنوعة من الحجر الجيري الموجودة في المتحف؟",
        "artifact_id": None
    },
    # Camera mode - English
    {
        "question": "What is this stela about and who is depicted on it?",
        "artifact_id": "Siptah Stela"
    },
    # Text mode - English
    {
        "question": "Tell me about the wives and queens of ancient Egyptian pharaohs represented here",
        "artifact_id": None
    },
]

@app.get("/benchmark")
def benchmark():
    results = []
    for test_question in test_questions:
        for model_label, model_name in MODELS.items():
            process = psutil.Process()
            gpu_before = get_gpu_stats()
            mem_before = process.memory_info().rss / 1024 / 1024

            start = time.time()
            result = ask_gem_guide(
                resources["index"],
                resources["chroma"],
                # resources["llm"],
                model_name,
                test_question['question'],
                test_question['artifact_id'],
                verbose=False
            )
            elapsed = time.time() - start

            gpu_after = get_gpu_stats()
            mem_after = process.memory_info().rss / 1024 / 1024

            results.append({
                "question": test_question['question'],
                "model": model_label,
                "time_seconds": round(elapsed, 2),
                "ram_delta_mb": round(mem_after - mem_before, 2),
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "vram_delta_mb": (gpu_after["vram_used_mb"] - gpu_before["vram_used_mb"]) if gpu_before and gpu_after else None,
                "response_length": len(result["raw_response"]),
                "response": result["raw_response"],
            })

    return {"results": results}

@app.post("/ask/stream")
def ask_guide_stream(request: QueryRequest):
    mode = "camera" if request.artifact_id else "text"
    
    # Retrieval
    if mode == "camera":
        retrieved_nodes = retrieve_by_artifact_id(
            resources["chroma"], resources["index"], request.artifact_id
        )
        if not retrieved_nodes:
            return {"result": "I couldn't find information about this artifact.", "sources": []}
    else:
        retriever = resources["index"].as_retriever(similarity_top_k=5)
        retrieved_nodes = retriever.retrieve(request.question)
    
    # Build prompt
    prompt = build_rag_prompt(request.question, retrieved_nodes)
    max_tokens = 670 if mode == "camera" else 800
    llm = get_llm("batiai/gemma4-e2b:q4", max_tokens)

    def generate():
        for chunk in llm.stream([
            SystemMessage(content=prompt),
            HumanMessage(content="Please answer the tourist's question based on the context provided.")
        ]):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")