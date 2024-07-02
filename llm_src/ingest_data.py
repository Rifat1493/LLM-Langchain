# import sys
# import os
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(parent_dir)
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader

from llm_src import utils, config


def create_web_data():

    if not os.path.exists(config.CACHE_FOLDER):
        os.makedirs(config.CACHE_FOLDER)

    collection_name = config.COLLECTION_NAME
    chroma_client = chromadb.PersistentClient(path=config.CACHE_FOLDER)
    # If you have created the collection before, you need to delete the collection first
    if len(chroma_client.list_collections()) > 0 and collection_name in [
        chroma_client.list_collections()[0].name
    ]:
        chroma_client.delete_collection(name=collection_name)

    print(f"Creating collection: '{collection_name}'")
    # chroma_client.create_collection(name=collection_name)

    # loader = WebBaseLoader(config.URL)
    # documents = loader.load()

    # loader = TextLoader(
    #     config.TEXT_URL,
    #     encoding="utf8",
    # )
    # documents = loader.load()

    pdf_loader = PyMuPDFLoader(config.PDF_URL)
    documents = pdf_loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)

    embeddings = utils.get_embeddings()

    # create the vectorestore to use as the index
    Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=config.CACHE_FOLDER,
    )

    print("Collection has been created successfully")



