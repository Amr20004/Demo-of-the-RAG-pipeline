import json
from llama_index.core import Document

def loading_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        chunks = json.load(file)

    documents = []
    for chunk in chunks:

        meta_text = ""
        if chunk.get('title'):
            meta_text += f"artifact label: {chunk['title']}. "
        if chunk.get('material'):
            meta_text += f"Material: {chunk['material']}. "
        if chunk.get('was_found_at'):
            meta_text += f"Found at: {chunk['was_found_at']}. "
        if chunk.get('historical_overview'):
            meta_text += f"Period: {chunk['historical_overview']}. "
        
        text = chunk['artifact_overview'].strip() if len(chunk['artifact_overview']) > 1 else ""
        enriched_text = f"{meta_text}\n{text}"

        doc = Document(
            text = enriched_text,
            metadata = {
                "title":chunk['title'],
                "historical_overview": chunk['historical_overview'],
                "material": chunk['material'],
                "was_found_at": chunk['was_found_at'],
                "width": chunk['width'],
                "length": chunk['length']
            }
        )
        documents.append(doc)

    print(f"there are {len(documents)} documents")
    return documents

# file_path = 'artifactDataset.json'
# documents = loading_dataset(file_path)
# print(documents[0])
