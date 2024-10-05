import streamlit as st
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("AIzaSyBJGzt-025A-JqVA68JEG-3untCz9EbQ38"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            # Try extracting text using pdfminer
            extracted_text = extract_text(pdf)
            if extracted_text.strip():  # Check if extracted text is not empty
                text += extracted_text + "\n"
            else:
                # Fallback to PyPDF2 if pdfminer extraction is empty
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
    return text.strip()


def get_text_chunks(text):
    if not text.strip():  # Ensure the text is not empty
        st.error("No valid text found for chunking.")
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error during text chunking: {str(e)}")
        return []


def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text chunks available for vector store creation.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        if os.path.exists("faiss_index"):
            try:
                vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.error(f"Error loading existing FAISS index: {str(e)}")
                return None
            progress_bar = st.progress(0)
            for i, chunk in enumerate(text_chunks):
                vector_store.add_texts([chunk])
                progress_bar.progress((i + 1) / len(text_chunks))
        else:
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating/updating vector store: {str(e)}")
        return None

def get_conversational_chain():
    prompt_template = """
    Using the provided context, answer the question as thoroughly and accurately as possible. If the answer is not available in the context, clearly state: "I'm sorry, but I don't have enough information to answer that question based on the provided context." Please do not make up or infer any information that is not directly supported by the context.

    Context: {context}

    Question: {question}

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error setting up the conversational chain: {str(e)}")
        return None

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search_with_score(user_question, k=5)

        if not docs:
            st.warning("No relevant information found in the uploaded documents.")
            return

        context = "\n".join([doc.page_content for doc, score in docs if score < 0.5])
        
        chain = get_conversational_chain()
        if chain is None:
            return

        response = chain(
            {"input_documents": [doc for doc, _ in docs], "question": user_question},
            return_only_outputs=True
        )

        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred while processing your question: {str(e)}")

def main():
    st.set_page_config("PDF Chatbot", page_icon="🤖", layout="wide")
    st.header("PDF Chatbot 🤖")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about the uploaded PDFs ✍️")

    if user_question:
        st.session_state.chat_history.append(("You", user_question))
        user_input(user_question)

    with st.sidebar:
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                if not pdf_docs:
                    st.error("Please upload at least one PDF file.")
                else:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("No text could be extracted from the uploaded PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = get_vector_store(text_chunks)
                        if vector_store is not None:
                            st.success("PDFs processed successfully!")
                        else:
                            st.error("Failed to process PDFs. Please try again.")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []

    st.subheader("Chat History")
    for role, message in st.session_state.chat_history:
        st.write(f"{role}: {message}")

if __name__ == "__main__":
    main()
