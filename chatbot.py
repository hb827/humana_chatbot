import os
from typing import Dict
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
import streamlit as st



def parse_retriever_input(params: Dict):
    return params["messages"][-1].content

@st.cache_resource 
def load_system():  
    try:  
        required_files = ['knowledge_base/knowledge_base.pkl', 'knowledge_base/knowledge_base.faiss', 'knowledge_base/model_name.txt']  
        for file in required_files:  
            if not os.path.exists(file):  
                print(f"Missing: {file}")  
                return None, None 
        with open('knowledge_base/model_name.txt', 'r') as f: 
            model_choice = f.read().strip() 
            model = HuggingFaceEmbeddings(model_name=model_choice) 
        index_name = "knowledge_base"
        index = FAISS.load_local(folder_path="knowledge_base/", embeddings=model, index_name=index_name,allow_dangerous_deserialization=True)
        return model, index 
    except Exception as e: 
        print(f"Error: {str(e)}") 
        return None, None 
    
def extract_answer(llm_response,initial_query):
    if "answer" in llm_response.keys():
        return llm_response["answer"].split(initial_query)[-1]
    else:
        print(llm_response)
        raise KeyError("The 'answer' key was not found in the retrieval chain output.")

def answer_question(vectorstore, query):
    hf_pipeline = pipeline("text-generation", model="gpt2")
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    retriever = vectorstore.as_retriever()

    SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context. 
    If the context doesn't contain any relevant information to the question, don't make something up. Instead, say "I couldn't find the answer to your question in the document".
    Do not repeat the context or the user's query in your response.
    
    <context>
    {context}
    </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

    retrieval_chain = RunnablePassthrough.assign(
        context=parse_retriever_input | retriever,
    ).assign(
        answer=document_chain
    )

    result = retrieval_chain.invoke(
        {
            "messages": [
                HumanMessage(content=query)
            ],
        }
    )


    ##############
    # below is to implement multi-turn


    #query_transform_prompt = ChatPromptTemplate.from_messages(
    #    [
    #        MessagesPlaceholder(variable_name="messages"),
    #        (
    #            "user",
    #            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
    #        ),
    #    ]
    #)

    #query_transforming_retriever_chain = RunnableBranch(
    #    (
    #        lambda x: len(x.get("messages", [])) == 1,
    #        # If only one message, then we just pass that message's content to retriever
    #        (lambda x: x["messages"][-1].content) | retriever,
    #    ),
    #    # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
    #    query_transform_prompt | llm | StrOutputParser() | retriever,
    #).with_config(run_name="chat_retriever_chain")


    #SYSTEM_TEMPLATE = """
    #Answer the user's questions based on the below context. 
    #If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":
    #Do not repeat the context or the user's query in your response.

    #<context>
    #{context}
    #</context>
    #"""

    #question_answering_prompt = ChatPromptTemplate.from_messages(
    #    [
    #        (
    #            "system",
    #            SYSTEM_TEMPLATE,
    #        ),
    #        MessagesPlaceholder(variable_name="messages"),
    #    ]
    #)

    #document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

    #conversational_retrieval_chain = RunnablePassthrough.assign(
    #    context=query_transforming_retriever_chain,
    #).assign(
    #    answer=document_chain,
    #)

    #result = conversational_retrieval_chain.invoke(
    #    {
    #        "messages": [
    #            HumanMessage(content=query),
    #        ]
    #    }
    #)

    ###########

    return result
