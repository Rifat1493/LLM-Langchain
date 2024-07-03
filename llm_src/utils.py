import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    pipeline,
)
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory

from llm_src import config

### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_embeddings():

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, cache_folder=config.CACHE_FOLDER
    )
    return embeddings


def get_llm_rag():
    rag_llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-large",
        task="text2text-generation",
        model_kwargs={
            "temperature": 0,
            "max_length": 1024,
            "cache_dir": config.CACHE_FOLDER,
        },
        device=0,
    )
    return rag_llm


def get_llm_agent():

    dotenv_path = Path("C:/Users/Rifat1493/Desktop/LLM-Langchain/dev.env")
    load_dotenv(dotenv_path=dotenv_path)
    # load_dotenv()
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    login(token=HUGGINGFACEHUB_API_TOKEN)

    model_name = "mistralai/Mistral-7B-v0.1"
    # model_config = transformers.AutoConfig.from_pretrained(
    #     model_name, cache_dir = config.CACHE_FOLDER
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=config.CACHE_FOLDER
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, cache_dir=config.CACHE_FOLDER
    )

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=100,
    )

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return mistral_llm


def extract_pandas_answer(response):
    splitted_response = response.split("\n\n")
    final_answer = None
    in_final_answer_section = False

    for item in splitted_response:
        if 'Begin' in item:
            in_final_answer_section = True
        if in_final_answer_section and 'Final Answer:' in item:
            final_answer = item.split('Final Answer: ')[-1]
            break

    return final_answer
