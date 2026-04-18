from langchain_ollama import ChatOllama

_models = {}

def get_llm(model_name: str = "batiai/gemma4-e2b:q4", num_predict: int = 800):
    global _models
    cache_key = f"{model_name}_{num_predict}"
    if cache_key not in _models:
        _models[cache_key] = ChatOllama(
            model=model_name,
            num_ctx=1024,
            num_predict=num_predict,
            temperature=0.3,
            repeat_penalty=1.1,
            keep_alive=-1,
        )
    return _models[cache_key]