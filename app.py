from load_dataset import loading_dataset
from index_dataset import indexing_dataset, load_existing_index
from Rag_engine import ask_gem_guide

file_path = 'artifactDataset.json'
# user_question = "tell me about this artifact, who is he?"
# artifact_id = "Vizier Khaemwaset"
documents = loading_dataset(file_path)
index, chroma_collection = indexing_dataset(documents)
# result = ask_gem_guide(index, chroma_collection, user_question, artifact_id)