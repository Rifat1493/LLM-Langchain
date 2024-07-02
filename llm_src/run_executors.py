import pandas as pd
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from llm_src import utils, config


def get_pandas_executor():
    df = pd.read_csv(config.CSV_DATA)
    # world_data
    pandas_executor = create_pandas_dataframe_agent(
        utils.get_llm_agent(), df, verbose=True
    )
    # tools = [PythonREPLTool()]
    # agent_executor = AgentExecutor(
    #     agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    # )

    return pandas_executor


def get_doc_executor():
    db_index = Chroma(
        config.COLLECTION_NAME,
        embedding_function=utils.get_embeddings(),
        persist_directory=config.CACHE_FOLDER,
    )
    # expose this index in a retriever interface
    retriever = db_index.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    doc_executor = RetrievalQA.from_chain_type(
        llm=utils.get_llm_rag(), chain_type="stuff", retriever=retriever
    )

    return doc_executor


def get_route_executor():

    route_executor = (
        PromptTemplate.from_template(
            """Given the user question below, classify it as either being about `Dataframe` or `Other`.

    Do not respond with more than one word.

    <question>
    {question}
    </question>

    Classification:"""
        )
        | utils.get_llm_rag()
        | StrOutputParser()
    )
    return route_executor


def get_doc_executor_history():
    print("Will be implemented soon")


def get_doc_streaming_executor():
    print("Will be implemented soon")
