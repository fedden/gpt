#!/usr/bin/env python3
import argparse

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp

import constants


def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDINGS_MODEL_NAME)
    db = Chroma(
        persist_directory=constants.PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=constants.CHROMA_SETTINGS,
    )
    retriever = db.as_retriever(search_kwargs={"k": constants.TARGET_SOURCE_CHUNKS})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match constants.MODEL_TYPE:
        case "LlamaCpp":
            llm = LlamaCpp(
                model_path=constants.MODEL_PATH,
                n_ctx=constants.MODEL_N_CTX,
                callbacks=callbacks,
                verbose=False,
            )
        case "GPT4All":
            llm = GPT4All(
                model=constants.MODEL_PATH,
                n_ctx=constants.MODEL_N_CTX,
                backend="gptj",
                callbacks=callbacks,
                verbose=False,
            )
        case _default:
            print(f"Model {constants.MODEL_TYPE} not supported!")
            exit
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not args.hide_source,
    )
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = (
            res["result"],
            [] if args.hide_source else res["source_documents"],
        )

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "privateGPT: Ask questions to your documents without an internet "
            "connection, using the power of LLMs."
        )
    )
    parser.add_argument(
        "--hide-source",
        "-S",
        action="store_true",
        help="Use this flag to disable printing of source documents used for answers.",
    )
    parser.add_argument(
        "--mute-stream",
        "-M",
        action="store_true",
        help="Use this flag to disable the streaming StdOut callback for LLMs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
