from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from  langchain_community.embeddings import HuggingFaceEmbeddings
import bs4
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser




class Chatbot():
    load_dotenv()
    loader = WebBaseLoader(
        web_path=("https://ntb.gov.np/ilam/",),
        bs_kwargs= dict(
            parse_only = bs4.SoupStrainer(
                class_ = ("tm-page")
            )          
        ),
    )
    
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 4)
    document = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=document , embedding= HuggingFaceEmbeddings())
    retriever = vectorstore.as_retriever()
    
    llm = Ollama(model= "llama3")
    
    template = """
        You are an tourist Guide. You will be asked about a place in Nepal called Ilam.
        You can use the above given Webloader to provide the relevant information.
        You must make sure to provide correct answer.
        Keep the answer within  3 sentences and concise.
        
        context = {context}
        Question: {question}
        Answer:
    
    """
    
    prompt = PromptTemplate(
        template = template,
        input_variables = ["context" , "question"]
    )
    
    rag_chain = (
        {"context": retriever , "question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        
    )
    

