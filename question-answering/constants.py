from chromadb.config import Settings

# The folder you want your vectorstore in
PERSIST_DIRECTORY: str = "db"
# supports LlamaCpp or GPT4All
MODEL_TYPE: str = "GPT4All"
# Path to your GPT4All or LlamaCpp supported LLM
MODEL_PATH: str = "ggml-gpt4all-j-v1.3-groovy.bin"
# SentenceTransformers embeddings model name (see https://www.sbert.net/docs/pretrained_models.html)
EMBEDDINGS_MODEL_NAME: str = "all-MiniLM-L6-v2"
# Maximum token limit for the LLM model
MODEL_N_CTX: int = 1000
# The amount of chunks (sources) that will be used to answer a question
TARGET_SOURCE_CHUNKS: int = 4
# Where the documents are.
SOURCE_DIRECTORY: str = "documents"
# Debugble for loops..
SINGLE_PROCESS: bool = True
# Chunks to be embedded.
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)
