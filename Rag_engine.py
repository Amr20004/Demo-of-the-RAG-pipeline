from llama_index.core.schema import TextNode, NodeWithScore
from load_model import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
import requests
# === RAG System Prompt ===


def detect_language(text: str) -> str:
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    return "arabic" if arabic_chars > len(text) * 0.3 else "english"


def build_rag_prompt(query: str, retrieved_contexts: list) -> str:
    """Build the prompt that combines retrieved context with the user's question."""

    # SYSTEM_PROMPT = """You are an expert Egyptology AI assistant serving as a tour guide at the 
    #     Grand Egyptian Museum (GEM) in Giza, Egypt. You help tourists understand ancient Egyptian 
    #     artifacts, pharaohs, and history.

    #     IMPORTANT RULES:
    #     1. ONLY use information from the provided context to answer questions.
    #     2. If the context doesn't contain enough information, say "I don't have specific information 
    #     about that in my knowledge base" rather than making up facts.
    #     3. Be engaging and educational, like a knowledgeable museum guide.
    #     4. When discussing artifacts, mention their materials, dimensions, and historical significance.
    #     5. When discussing pharaohs, include their dynasty, reign dates, and key achievements.

    # - Use EXACT location names from the context. Never guess or translate location names.
        # - Always provide a comprehensive answer. Include specific examples, names, and details from the context. Never give a response shorter than 3 sentences.
    # """

    SYSTEM_PROMPT = """You are an Egyptology tour guide at the Grand Egyptian Museum (GEM).
        Rules:
        - ONLY use the provided context. If insufficient, say so.
        - Be engaging and educational.
        - Mention materials, dimensions, location, and historical significance.
        - Include dynasty and reign dates for pharaohs.
        - Use EXACT location names from the context. Never guess or translate location names.
        - Always provide a comprehensive answer. Include specific examples, names, and details from the context. Never give a response shorter than 3 sentences.
    """

    language = detect_language(query)

    if language == "arabic":
        language_instruction = "CRITICAL: The tourist is speaking Arabic. You MUST respond ENTIRELY in Arabic."
    else:
        language_instruction = "The tourist is speaking English. Respond in English."
    
    # Format retrieved contexts
    context_parts = []
    for i, node in enumerate(retrieved_contexts, 1):
        meta = node.metadata
        meta_str = ""
        if meta.get('width'):
            meta_str += f"Width: {meta['width']}\n"
        if meta.get('length'):
            meta_str += f"Length: {meta['length']}\n"
    
        # text = node.text[:500] if len(node.text) > 500 else node.text
        context_parts.append(f"[{i}] {meta_str}{node.text}")
    
    context_block = "\n".join(context_parts)
    
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{language_instruction}\n\n"
        f"=== RETRIEVED KNOWLEDGE BASE CONTEXT ===\n"
        f"{context_block}\n"
        f"=== END OF CONTEXT ===\n\n"
        f"Tourist's Question: {query}\n\n"
    )
    
    return prompt


# def generate_response_direct(prompt: str, model_name: str) -> str:
#     response = requests.post(
#         "http://localhost:11434/api/chat",
#         json={
#             "model": model_name,
#             "messages": [
#                 {"role": "system", "content": prompt},
#                 {"role": "user", "content": "Please answer the tourist's question based on the context provided."}
#             ],
#             "stream": False,
#             "think": False,
#             "options": {
#                 "num_ctx": 2048,
#                 "num_predict": 800,
#                 "temperature": 0.3,
#                 "repeat_penalty": 1.1,
#             }
#         }
#     )
#     data = response.json()
#     content = data.get("message", {}).get("content", "")
#     print(f"[DEBUG] Model: {model_name}")
#     print(f"[DEBUG] Content: {repr(content[:200]) if content else 'EMPTY'}")
#     return content


def generate_response(prompt: str, model_name, max_new_tokens: int = 800) -> str:
    """Generate a response from the fine-tuned Gemma3n model."""
    # if "qwen" in model_name:
    #     return generate_response_direct(prompt, model_name)
    
    llm = get_llm(model_name, max_new_tokens)
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Please answer the tourist's question based on the context provided.")
    ])
    return response.content

def get_artifact_metadata(chroma_collection, artifact_id: str):
    try:
        artifact = chroma_collection.get(where={'title':artifact_id}, include=["documents","metadatas"])
        if not artifact['metadatas']:
            return None
        metadata = artifact['metadatas'][0]
        title = metadata['title']
        historical  = metadata['historical_overview']
        material  = metadata['material']
        width  = metadata['width']
        length  = metadata['length']
        was_found_at = metadata['was_found_at']
        structured_output = {
            'artifact_title': title,
            'artifact_historical_overview': historical,
            'artifact_material': material,
            'artifact_width': width,
            'artifact_length': length,
            'artifact_was_found_at': was_found_at,
        }
        return structured_output
        
    except Exception as e:
        print(f"  Metadata filter failed: {e}")



def retrieve_by_artifact_id(chroma_collection, index, artifact_id: str, top_k: int = 3):
    try:
        results = chroma_collection.get(where={"title":artifact_id}, include=["documents","metadatas"])
        if results and results["documents"]:
            retrieved_nodes = []
            for doc_text, metadata in zip(results["documents"], results["metadatas"]):
                node = TextNode(text=doc_text, metadata=metadata)
                nodeWithScore = NodeWithScore(node=node, score=1.0)
                retrieved_nodes.append(nodeWithScore)
            return retrieved_nodes[:top_k]
    except Exception as e:
        print(f"  Metadata filter failed: {e}, falling back to similarity search")

    # Fallback use the artifact_id as a similarity query
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(artifact_id)
    


def ask_gem_guide(index, chroma_collection, model_name, question: str, artifact_id: str = None, top_k: int = 5, verbose: bool = True) -> dict:
    mode = "camera" if artifact_id else "text"
    
    if verbose:
        print(f"\n{'='*60}")
        if mode == "camera":
            print(f" Mode: CAMERA (artifact identified)")
            print(f" Artifact ID: {artifact_id}")
            print(f" Tourist's Question: {question}")
        else:
            print(f" Mode: TEXT (no camera)")
            print(f" Tourist's Question: {question}")
        print(f"{'='*60}")

    if mode == "camera":
        retrieved_nodes = retrieve_by_artifact_id(chroma_collection, index, artifact_id)   
        if not retrieved_nodes:
            return {
                "question": question,
                "artifact_id": artifact_id,
                "mode": mode,
                "raw_response": "I couldn't find information about this artifact in my knowledge base. Please try again or ask a general question.",
                "sources": [],
            }
        if verbose:
            print(f"\n Retrieved {len(retrieved_nodes)} chunks (by artifact ID):")
            for i, node in enumerate(retrieved_nodes, 1):
                title = node.metadata.get('title', 'N/A')[:50]
                print(f"   {i}. {title} — Score: {node.score:.4f}")
    
    else:
        retriever = index.as_retriever(similarity_top_k=top_k)
        retrieved_nodes = retriever.retrieve(question)
        if verbose:
            print(f"\n Retrieved {len(retrieved_nodes)} chunks:")
            for i, node in enumerate(retrieved_nodes, 1):
                print(f"   {i}. {node.metadata.get('title', 'N/A')[:50]} — Score: {node.score:.4f}")

    
    # Step 2: Build the RAG prompt
    prompt = build_rag_prompt(question, retrieved_nodes)
    if verbose:
        print(f"\n Prompt length: {len(prompt)} characters")

    max_new_tokens = 670 if mode == "camera" else 800 

    # Step 3: Generate response
    if verbose:
        print(f"\n Generating response...")
    raw_response = generate_response(prompt, model_name, max_new_tokens)

    # fallback if the response is lower than 250 char
    # if mode == "text" and len(raw_response) < 300:
    #     raw_response = generate_response(prompt, model_name, max_new_tokens)

    if verbose:
        print(f"\n AI Guide Response:\n{raw_response}")

    
    # Step 4: Return structured result
    sources = [
        {
            "title": node.metadata.get('title', 'N/A'),
            "score": float(node.score),
        }
        for node in retrieved_nodes
    ]
    
    return {
        "question": question,
        "artifact_id": artifact_id,
        "mode": mode,
        # "answer": parsed,
        "raw_response": raw_response,
        "sources": sources,
    }