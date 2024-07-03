import pandas as pd
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
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
    db_index = Chroma(
        config.COLLECTION_NAME,
        embedding_function=utils.get_embeddings(),
        persist_directory=config.CACHE_FOLDER,
    )
    # expose this index in a retriever interface
    retriever = db_index.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        utils.get_llm_rag(), retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(utils.get_llm_rag(), qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_executor = RunnableWithMessageHistory(
        rag_chain,
        utils.get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_executor


def get_doc_streaming_executor():
    print("Will be implemented soon")
