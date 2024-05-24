from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]= 'true'
os.environ["]LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , "You are a very good"),
        ("user" , "Question:{question}")
    ]
)

st.title("Langchain demo for OLLAMA")
input_text = st.text_input("Search")

llm = Ollama(model = "llama3")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))

