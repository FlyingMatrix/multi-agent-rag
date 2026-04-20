from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from settings import Settings as AppSettings 

def build_embed_model(app_settings: AppSettings):
    # Create the local embedding model
    embed_model = HuggingFaceEmbedding(
        model_name=app_settings.embed_model_name
    )
    Settings.embed_model = embed_model
    Settings.llm = None     # let LlamaIndex explicitly not to initialize a default LLM
    
    return embed_model
