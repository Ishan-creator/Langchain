from main import Chatbot
import streamlit as st

bot = Chatbot()

st.set_page_config(page_title="Ask me About Ilam")
with st.sidebar:
    st.title("Ask me about Ilam")
    
    
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return result

if "messagges" not in st.session_state.keys():
    st.session_state.messages = [{"role":"assistant" , "content":"Welcome to Ilam"}]
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)
        
        
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer from mystery stuff.."):
            response = generate_response(input) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

