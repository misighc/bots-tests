import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate


openai.api_key = st.secrets["OPENAI_API_KEY"]
@st.cache_resource

def load_chain():

    """
The `load_chain()` function initializes and configures a conversational retrieval chain for
answering user questions.
:return: The `load_chain()` function returns a ConversationalRetrievalChain object.
"""

    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings()

    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0.5,
                     model="gpt-3.5-turbo-1106",
                     streaming=True,
                     max_tokens=1000,
                     verbose=True,
                     request_timeout=60,
                     )

    # Load our local FAISS index as a retriever
    vector_store = FAISS.load_local("faiss_index1", embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Create memory 'chat_history'
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

    # Create system prompt
    template = """
Welcome! I am your AI assistant, equipped to assist you with inquiries about the document titled "Currículo Base da Educação Infantil e Ensino Fundamental do Território Catarinense." This document serves as a comprehensive resource detailing the educational curriculum within the region.

Feel free to pose questions pertaining to the content of this document, and I will furnish you with detailed and informative responses based on the wealth of information it contains. Should you find yourself uncertain about a particular aspect, don't hesitate to ask—I'll promptly inform you if the required information is unavailable. It's important to keep your queries within the confines of this document's scope for optimal assistance.

For instance, you can delve into specific curriculum components, educational objectives, or any other pertinent topics covered in the document. This ensures that the responses provided align closely with the document's content.

Keep in mind, if I lack the necessary information to address your question accurately, I'll express my limitations with a response like 'Sorry, I don't know... 😔.' This commitment to honesty ensures the reliability and precision of the information shared.

Feel free to start exploring the wealth of knowledge contained within the document by asking questions aligned with its content.   
    {context}
    Question: {question}
    Helpful Answer:"""

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                  retriever=retriever,
                                                  memory=memory,
                                                  get_chat_history=lambda h: h,
                                                  verbose=True)

    # Add systemp prompt to chain
    # Can only add it at the end for ConversationalRetrievalChain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)

    return chain
