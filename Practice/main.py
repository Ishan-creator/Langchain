import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import bs4

# Load and process the web page
st.write("Loading the web page and processing documents...")
loader = WebBaseLoader(
    web_path=("https://ntb.gov.np/ilam/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("tm-page")
        )
    ),
)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

# Create vectorstore and retriever
vectorstore = Chroma.from_documents(documents=split_docs, embedding=OllamaEmbeddings(model="llama3"))
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | Ollama(model="llama3")
    | StrOutputParser()
)

# Streamlit interface
st.title("Ilam Information Retrieval")
st.write("Ask questions about Ilam based on the content from the specified web page.")

user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Retrieving information..."):
        response = rag_chain.invoke(user_question)
    st.write("Response:")
    st.write(response)
