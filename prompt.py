import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Qdrant
import random
from datetime import datetime
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import string
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document

def get_qa_chain(vectorstore,num_chunks):

    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-3.5-turbo"), chain_type="stuff", retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks}),  return_source_documents=True)
    return qa