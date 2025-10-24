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

# =====================================================
# STEP 2: PDF PROCESSING FUNCTION
# =====================================================

def process_pdf(pdf_file):
    """Process PDF and create searchable vector database."""
    global vectorstore

    try:
        if not pdf_file:
            return "⚠️ Please upload a PDF file first!"

        if not pdf_file.name.lower().endswith('.pdf'):
            return "⚠️ Invalid file! Please upload a PDF file."

        logger.info(f"📄 Processing: {pdf_file.name}")

        # Load PDF
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()

        if not documents:
            return "⚠️ No text found in PDF!"

        logger.info(f"✅ Loaded {len(documents)} pages")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        logger.info(f"✅ Created {len(chunks)} chunks")

        # Create FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")

        logger.info("✅ FAISS index created!")

        return f"✅ Success! Processed {len(documents)} pages into {len(chunks)} chunks!"

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return f"❌ Error: {str(e)}"

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def answer_query(query):
    """Answer user's questions using RAG approach."""
    global vectorstore

    if vectorstore is None:
        return "⚠️ Please upload and process a PDF first!"

    if not query.strip():
        return "⚠️ Please enter a question!"

    try:
        logger.info(f"❓ Question: {query}")

        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant answering questions based on the provided documents.

Use the following context to answer the question.
If the answer is not in the context, say "I cannot find this in the document"
and provide a general answer.

Context:
{context}

Question: {question}

Answer:""")

        # Set up retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )

        #Create LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        #Build RAG chain
        rag_chain = (
            {"context": retriever | format_docs,"question":RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Get answer
        answer = rag_chain.invoke(query)

        # Get source
        source_docs = retriever.invoke(query)
        sources = format_sources(source_docs)

        logger.info(f"✅ Answer generated!")
        return answer, sources
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return f"❌ Error: {str(e)}"

def format_sources(source_docs):
    """Format source documents for display."""
    if not source_docs:
        return "No sources found."

    source_text = "📚 **Sources Used:**\n\n"

    for i, doc in enumerate(source_docs, 1):
        page = doc.metadata.get("page", "Unknown")
        preview = doc.page_content[:200].replace('\n', ' ').strip()
        source_text += f"**Source {i}** (Page {page}):\n{preview}...\n\n"

    return source_text

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 📚 RAG-Based PDF Chatbot
    Upload a PDF and ask questions about it!
    """)

    # PDF Upload Section
    gr.Markdown("## 📤 Step 1: Upload PDF")
    with gr.Row():
        pdf_input = gr.File(label="📄 Choose PDF", file_types=[".pdf"])
        process_btn = gr.Button("🔄 Process PDF", variant="primary")

    status_output = gr.Textbox(label="📊 Status", interactive=False, lines=2)

    # Q&A Section
    gr.Markdown("## ❓ Step 2: Ask Questions")
    with gr.Row():
        query_input = gr.Textbox(
            label="💭 Your Question",
            placeholder="What would you like to know?",
            lines=2
        )
        query_btn = gr.Button("🤖 Get Answer", variant="primary")

    answer_output = gr.Textbox(label="💡 Answer", interactive=False, lines=8)
    sources_output = gr.Textbox(label="📚 Sources", interactive=False, lines=6)

    # Wire up events
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[status_output]
    )

    query_btn.click(
        fn=answer_query,
        inputs=[query_input],
        outputs=[answer_output, sources_output]
    )

    query_input.submit(
        fn=answer_query,
        inputs=[query_input],
        outputs=[answer_output, sources_output]
    )


# Launch the app
if __name__ == "__main__":
    logger.info("🚀 Launching Gradio interface...")
    demo.launch(share=True)
