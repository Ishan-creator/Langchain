from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description = "API server"
)

add_routes(
    app, 
    Ollama(),
    path ="/llama"
)
llm = Ollama(model = "llama3")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about{topic} with 50 words")
prompt2 = ChatPromptTemplate.from_template("Write me an poem about{topic} with 50 words")

add_routes(
    app,
    prompt1|llm,
    path = "/essay"
)
add_routes(
    app,
    prompt2|llm,
    path = "/poem"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)
