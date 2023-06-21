from chromadb.config import Settings

# The folder you want your vectorstore in
PERSIST_DIRECTORY: str = "db"
# supports LlamaCpp, GPT4All, HuggingFaceHub, HuggingFacePipeline
MODEL_TYPE: str = "HuggingFacePipeline" # "GPT4All"
# Path to your GPT4All or LlamaCpp supported LLM
MODEL_PATH: str = "models/ggml-gpt4all-j-v1.3-groovy.bin"
# Token.
HUGGINGFACE_HUB_API_TOKEN: str = ""
# SentenceTransformers embeddings model name (see https://www.sbert.net/docs/pretrained_models.html)
EMBEDDINGS_MODEL_NAME: str = "all-MiniLM-L6-v2"
# Maximum token limit for the LLM model
MODEL_N_CTX: int = 1000
# The amount of chunks (sources) that will be used to answer a question
TARGET_SOURCE_CHUNKS: int = 10
# Where the documents are.
SOURCE_DIRECTORY: str = "documents"
# Debugble for loops..
SINGLE_PROCESS: bool = True
# Chunks to be embedded.
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100
# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)
USE_PROMPT: bool = True
