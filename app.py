import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import requests
from typing import List
import easyocr
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader
)

# Configuration
load_dotenv()
GATEWAY_URL = os.getenv("GATEWAY_ENDPOINT_URL")
API_KEY = os.getenv("OPENAI_KEY")


class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                f"{GATEWAY_URL}/embeddings",
                headers=headers,
                json={"model": "text-embedding-ada-002", "input": texts},
                verify=False
            )
            response.raise_for_status()
            return [item['embedding'] for item in response.json()['data']]
        except requests.exceptions.HTTPError as e:
            if response.status_code == 403:
                st.error("Authentication failed. Please check your API token.")
            else:
                st.error(f"Error: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return []

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class CustomChatOpenAI:
    def invoke(self, message: str) -> str:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4-turbo",  # Any model which your LLM API supports
            "messages": [{"role": "user", "content": message}]
        }
        response = requests.post(
            f"{GATEWAY_URL}/chat/completions",
            headers=headers,
            json=data,
            verify=False
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code}"


def load_document(uploaded_file):
    """Load document from uploaded file"""
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Handle image files with EasyOCR
        if file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff']:
            try:
                reader = easyocr.Reader(['en'])
                results = reader.readtext(tmp_file_path)
                extracted_text = '\n'.join([result[1] for result in results])

                if extracted_text.strip():
                    documents = [Document(
                        page_content=extracted_text,
                        metadata={"source": uploaded_file.name, "type": "image"}
                    )]
                else:
                    documents = [Document(
                        page_content=f"Image file: {uploaded_file.name}\nNo text could be extracted from this image.",
                        metadata={"source": uploaded_file.name, "type": "image"}
                    )]
            except Exception as e:
                st.error(f"Error processing image with OCR: {e}")
                documents = [Document(
                    page_content=f"Image file:{uploaded_file.name}\nError extracting text from image.",
                    metadata={"source": uploaded_file.name, "type": "image"}
                )]
        else:
            try:
                if file_extension == 'pdf':
                    loader = PyPDFLoader(tmp_file_path)
                elif file_extension == 'txt':
                    loader = TextLoader(tmp_file_path)
                elif file_extension == 'csv':
                    loader = CSVLoader(tmp_file_path)
                elif file_extension in ['docx', 'doc']:
                    try:
                        loader = UnstructuredWordDocumentLoader(tmp_file_path)
                    except ImportError:
                        st.error("Missing dependency for Word documents. Install with : pip install python-docx")
                        os.unlink(tmp_file_path)
                        return []
                elif file_extension in ['pptx', 'ppt']:
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(tmp_file_path)
                        text_content = ""
                        for page in doc:
                            text_content += page.get_text() + "\n"
                        doc.close()

                        if text_content.strip():
                            documents = [Document(
                                page_content=text_content,
                                metadata={"source": uploaded_file.name, "type": "powerpoint"}
                            )]
                        else:
                            documents = [Document(
                                page_content=f"PowerPoint file: {uploaded_file.name}\nNo text could be extracted.",
                                metadata={"source": uploaded_file.name, "type": "powerpoint"}
                            )]
                    except ImportError:
                        st.error("Missing PyMuPDF. Install with: pip install PyMuPDF")
                        os.unlink(tmp_file_path)
                        return []

                    os.unlink(tmp_file_path)
                    return documents
                elif file_extension in ['xlsx', 'xls']:
                    content = ""
                    success = False

                    # Method 1: Try as CSV
                    try:
                        df = pd.read_csv(tmp_file_path)
                        content = df.to_string(index=False, na_rep='')
                        success = True
                        st.info(f"File '{uploaded_file.name}' read as CSV format")
                    except:
                        pass

                    # Method 2: Try Excel with different engines
                    if not success:
                        for engine in ['openpyxl', 'xlrd', None]:
                            try:
                                df_dict = pd.read_excel(tmp_file_path, sheet_name=None, engine=engine)
                                for sheet_name, df in df_dict.items():
                                    content += f"Sheet: {sheet_name}\n"
                                    content += df.to_string(index=False, na_rep='') + "\n\n"
                                success = True
                                break
                            except:
                                continue

                    # Method 3: Try as text file
                    if not success:
                        try:
                            with open(tmp_file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            success = True
                            st.info(f"File '{uploaded_file.name}' read as text file")
                        except:
                            try:
                                with open(tmp_file_path, 'r', encoding='latin-1') as f:
                                    content = f.read()
                                success = True
                                st.info(f"File '{uploaded_file.name}' read as text file with latin-1 encoding")
                            except:
                                pass

                    if success and content.strip():
                        documents = [Document(
                            page_content=content,
                            metadata={"source": uploaded_file.name, "type": "excel"}
                        )]
                    else:
                        st.error(f"Could not read file '{uploaded_file.name}'. File may be corrupted or in an unsupported format.")
                        documents = []
                        os.unlink(tmp_file_path)
                        return documents
                else:
                    st.error(f"Unsupported file type: {file_extension}")
                    os.unlink(tmp_file_path)
                    return []

                documents = loader.load()
            except Exception as e:
                st.error(f"Error loading {file_extension} file: {e}")
                os.unlink(tmp_file_path)
                return []

        os.unlink(tmp_file_path)
        return documents
    except Exception as e:
        st.error(f"Error loading file : {e}")
        os.unlink(tmp_file_path)
        return []


def chunk_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 200):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        chunked_docs.extend(chunks)

    return chunked_docs


def create_vectorstore(chunked_docs):
    """Create vector database from chunked documents"""
    embeddings = CustomEmbeddings()

    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db",
            collection_name="uploaded_docs"
        )
        vectorstore.add_documents(chunked_docs)
    else:
        vectorstore = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            persist_directory="./chroma_db",
            collection_name="uploaded_docs"
        )
    return vectorstore


def reset_database():
    """Completely reset the vector database and all data"""
    import shutil
    import time

    if 'vectorstore' in st.session_state and st.session_state.vectorstore is not None:
        try:
            st.session_state.vectorstore.delete_collection()
            st.session_state.vectorstore._client.reset()
            del st.session_state.vectorstore
        except:
            pass

    keys_to_keep = ['page_loaded']
    keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
    for key in keys_to_delete:
        del st.session_state[key]

    st.session_state.vectorstore = None
    st.session_state.processed = False

    if os.path.exists("./chroma_db"):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                shutil.rmtree("./chroma_db", ignore_errors=True)
                if not os.path.exists("./chroma_db"):
                    break
            except (PermissionError, OSError):
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                else:
                    try:
                        for root, dirs, files in os.walk("./chroma_db", topdown=False):
                            for file in files:
                                os.remove(os.path.join(root, file))
                            for dir in dirs:
                                os.rmdir(os.path.join(root, dir))
                        os.rmdir("./chroma_db")
                    except:
                        if 'page_loaded' in st.session_state:
                            st.warning("Database files may still exist. Please restart the application.")
                        return False
    return True


def query_documents(vectorstore, query: str, k: int = 3):
    """Query the vector database and generate response"""
    relevant_docs = vectorstore.max_marginal_relevance_search(
        query, k=k, fetch_k=k * 2, lambda_mult=0.5
    )

    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k * 2)

    doc_scores = {}
    for doc, score in docs_with_scores:
        doc_key = doc.page_content[:100]
        doc_scores[doc_key] = 1 - score

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    llm = CustomChatOpenAI()
    prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    return response, relevant_docs, doc_scores


# Streamlit App
st.title("Document Q&A RAG Application")
st.write("Upload any document and ask questions about its content!")

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'page_loaded' not in st.session_state:
    reset_database()
    st.session_state.page_loaded = True

uploaded_file = st.file_uploader(
    "Choose a file",
    type=['pdf', 'txt', 'csv', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls', 'jpg', 'jpeg', 'png', 'gif'],
    key="file_uploader"
)

if uploaded_file is not None:
    st.write(f"**File:** {uploaded_file.name}")

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Process Document")
    with col2:
        reset_btn = st.button("Reset Database")

    if reset_btn:
        if reset_database():
            st.success("Database reset successfully!")
        st.rerun()

    if process_btn:
        with st.spinner("Processing document..."):
            documents = load_document(uploaded_file)

            if documents:
                st.success(f"Loaded {len(documents)} pages/sections")
                chunked_docs = chunk_documents(documents)
                st.success(f"Created {len(chunked_docs)} chunks")
                st.session_state.vectorstore = create_vectorstore(chunked_docs)
                st.session_state.processed = True
                st.success("Document processed and stored in vector database!")

if st.session_state.processed and st.session_state.vectorstore:
    st.subheader("Ask Questions")
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Searching and generating answer..."):
            response, relevant_docs, doc_scores = query_documents(st.session_state.vectorstore, query)

            st.subheader("Answer:")
            st.write(response)

            st.subheader("Relevant Document Sections:")
            for i, doc in enumerate(relevant_docs, 1):
                doc_key = doc.page_content[:100]
                similarity_score = doc_scores.get(doc_key, 0.0)
                accuracy_percent = round(similarity_score * 100, 1)

                with st.expander(f"Source {i} - Relevance: {accuracy_percent}%"):
                    st.write(doc.page_content)

st.sidebar.title("Instructions")
st.sidebar.write("""
1. Upload a document (PDF, Word, Excel, PowerPoint, Text, CSV, or Image)
2. Click 'Process Document' to extract and index the content
3. Ask questions about the document content
4. Get AI-powered answers with source references
""")

st.sidebar.title("Supported File Types")
st.sidebar.write("""
- PDF (.pdf)
- Text (.txt)
- Word (.docx, .doc)
- PowerPoint (.pptx, .ppt)
- CSV (.csv)
- Excel (.xlsx, .xls)
- Images (.jpg, .jpeg, .png, .gif)                                  
""")
