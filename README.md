# 📄 DocumentQA – AI-Powered Document Question Answering

DocumentQA is a Streamlit-based application that allows you to upload documents (**PDF, DOCX, TXT, CSV, Images, etc.**) and interact with them using natural language queries.  
It uses **LangChain**, **ChromaDB**, and **OpenAI GPT models** to provide accurate answers based on the content of your uploaded files.

---

## 🚀 Features

- **Multiple File Format Support** – PDF, DOCX, TXT, CSV, JPG, PNG, and more.
- **Document Processing** – Extracts and embeds document content for semantic search.
- **AI-Powered Q&A** – Ask questions and get precise answers from your uploaded documents.
- **Persistent Vector Storage** – Uses ChromaDB to store embeddings locally.
- **Data Reset Option** – Delete all stored data with one click.
- **User-Friendly UI** – Built with Streamlit for an interactive experience.

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Streamlit** – Frontend web interface
- **LangChain** – Document loading, splitting, and embeddings
- **ChromaDB** – Vector database for document retrieval
- **OpenAI API** – LLM-based question answering
- **Pandas** – CSV & data handling
- **Pillow** – Image handling

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/sharuk08/DocumentQA.git
cd DocumentQA
```

### 2. Create a virtual environment (optional but recommended)
```bash
conda create -n documentqa python=3.10 -y
conda activate documentqa
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API Key
Create a `.env` file in the project folder with the following content:
```env
OPENAI_API_KEY=your_api_key_here
GATEWAY_ENDPOINT_URL=your_endpoint_gateway_url
```

---

## ▶ Usage

1. **Run the Streamlit app**
    ```bash
    streamlit run app.py
    ```
2. **Upload your documents** using the sidebar.
3. **Click "Process Document"** to embed and store the content.
4. **Ask your question** in the chat box.
5. Optionally, **use "Delete all the data"** to clear stored embeddings.

---

## 📁 Project Structure

```
DocumentQA/
│── app.py                # Main Streamlit application
│── requirements.txt      # Python dependencies
│── .env                  # Environment variables
│── chroma_db/            # Local vector database
│── README.md             # Project documentation
```

---

## 🔮 Future Improvements

- Support for audio & video transcription
- Cloud-based vector storage
- Multi-document conversational memory

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use, modify, and share it.
