from langchain_ollama import OllamaEmbeddings
import os

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

text_to_embed = "LangChain is a framework for developing applications powered by large language models (LLMs)."

single_vector = embeddings.embed_query(text_to_embed)
print(str(single_vector))