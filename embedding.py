from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from settings import Settings

def build_embed_model(settings: Settings):
    return HuggingFaceEmbedding(
        model_name=settings.embed_model_name
    )
