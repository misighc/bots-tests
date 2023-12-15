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
"Welcome! I am your AI assistant, designed to provide information and assist in various ways regarding the document titled "Currículo Base da Educação Infantil e Ensino Fundamental do Território Catarinense." This document serves as a comprehensive resource detailing the educational curriculum within the region.

Feel free to ask me questions about the content of this document, and I'll offer detailed and informative responses. In addition to answering questions, I can assist you in creating educational content and activities based on the document's information.

If you're looking to develop educational materials, simply provide me with specific details or themes you'd like to focus on. Whether it's generating textual content, crafting lesson plans, or suggesting interactive activities, I'm here to help.

For example, you can ask me to create a summary of a specific curriculum component, generate a set of quiz questions related to educational objectives, or propose engaging activities inspired by the document's content.

Remember, my goal is to be a versatile educational assistant, so don't hesitate to explore the various ways I can support you in enhancing educational experiences using the information from this document.

Should you have any questions or if there's a specific type of educational content you'd like assistance with, feel free to let me know, and we can embark on this educational journey together!"   
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
