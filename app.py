import streamlit as st
import os
from chatbot import load_system
from chatbot import answer_question
from chatbot import extract_answer
#from dotenv import load_dotenv

st.set_page_config(page_title="Slamon et al. 1987 Q&A", page_icon="💬")

#load_dotenv()
#os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("huggingface_api")

# 

def main():
    st.title("RAG Q&A App")

    model, index = load_system() 
    if not all([model, index]): 
        st.stop() 
    if "messages" not in st.session_state: 
        st.session_state.messages = [] 

    st.session_state.vectorstore = index
    st.success("PDF processed successfully. You can now ask questions.")

    if "vectorstore" in st.session_state:
        # Display chat 
        for message in st.session_state.messages: 
            with st.chat_message(message["role"]): 
                st.write(message["content"]) 
        # Chat input 
        if prompt := st.chat_input("Ask me about Slamon et al. (1987)"): 
            st.session_state.messages.append({"role": "user", "content": prompt}) 
        
            with st.chat_message("user"): 
                st.write(prompt) 
        
            with st.chat_message("assistant"): 
                response = extract_answer(answer_question(st.session_state.vectorstore, prompt),prompt) #.split("Answer:")[-1].strip()
                st.write(response) 
        
                st.session_state.messages.append({ 
                    "role": "assistant", 
                    "content": response
                }) 


if __name__ == "__main__":
    main()
