#!/usr/bin/env python3
import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

import constants


# Custom document loaders
class CustomEmailLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (CustomEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> Document:
    """Load document."""
    ext: str = os.path.splitext(file_path)[1]
    assert ext in LOADER_MAPPING, f"Unsupported file extension '{ext}'"
    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    os.environ["NLTK_DATA"] = "/home/jovyan/nltk_data/"
    try:
        return loader.load()[0]
    except:
        breakpoint()
        loader.load()


def get_filtered_files(source_dir: str, ignored_files: List[str]) -> List[str]:
    """Get all files."""
    assert os.path.isdir(source_dir), f"'{source_dir}' is not a directory"
    all_files: List[str] = []
    for ext in LOADER_MAPPING.keys():
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    ignored_files_set = set(ignored_files)
    filtered_files: List[str] = [
        file_path for file_path in all_files if file_path not in ignored_files_set
    ]
    assert (
        filtered_files
    ), f"0 `filtered_files` found in all of {len(all_files)} `all_files`"
    return filtered_files


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    filtered_file_paths: List[str] = get_filtered_files(
        source_dir=source_dir, ignored_files=ignored_files
    )
    results: List[Document] = []
    with tqdm(
        total=len(filtered_file_paths), desc="Loading new documents", ncols=80
    ) as pbar:
        if constants.SINGLE_PROCESS:
            for file_path in filtered_file_paths:
                doc: Document = load_single_document(file_path)
                results.append(doc)
                pbar.update()
        else:
            with Pool(processes=os.cpu_count()) as pool:
                for doc in pool.imap_unordered(
                    load_single_document, filtered_file_paths
                ):
                    results.append(doc)
                    pbar.update()
    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {constants.SOURCE_DIRECTORY}")
    documents = load_documents(constants.SOURCE_DIRECTORY, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {constants.SOURCE_DIRECTORY}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=constants.CHUNK_SIZE, chunk_overlap=constants.CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    print(
        f"Split into {len(texts)} chunks of text (max. {constants.CHUNK_SIZE} "
        f"tokens each)"
    )
    return texts


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, "index")):
        if os.path.exists(
            os.path.join(persist_directory, "chroma-collections.parquet")
        ) and os.path.exists(
            os.path.join(persist_directory, "chroma-embeddings.parquet")
        ):
            list_index_files = glob.glob(os.path.join(persist_directory, "index/*.bin"))
            list_index_files += glob.glob(
                os.path.join(persist_directory, "index/*.pkl")
            )
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


def main():
    # Create embeddings
    embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
        model_name=constants.EMBEDDINGS_MODEL_NAME
    )
    if does_vectorstore_exist(persist_directory=constants.PERSIST_DIRECTORY):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {constants.PERSIST_DIRECTORY}")
        db = Chroma(
            persist_directory=constants.PERSIST_DIRECTORY,
            embedding_function=embeddings,
            client_settings=constants.CHROMA_SETTINGS,
        )
        collection = db.get()
        texts = process_documents(
            [metadata["source"] for metadata in collection["metadatas"]]
        )
        print("Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print("Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=constants.PERSIST_DIRECTORY,
            client_settings=constants.CHROMA_SETTINGS,
        )
    db.persist()
    db = None
    print("Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
