import os
import logging
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

vectorstore = None

if not os.getenv("OPENAI_API_KEY"):
    logger.error("❌ OPENAI_API_KEY not found!")
    raise ValueError("OpenAI API key required!")
else:
    logger.info("✅ OpenAI API key loaded!")

logger.info("✅ Imports completed!")
