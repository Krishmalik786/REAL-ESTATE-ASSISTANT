from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

persist_dir = Path("resources/vectorstore")

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Vector DB
vectorstore = Chroma(
    collection_name="real_estate",
    embedding_function=embeddings,
    persist_directory=str(persist_dir)
)


def ingest(urls):
    loader = SeleniumURLLoader(urls=urls)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE)
    docs = splitter.split_documents(data)

    vectorstore.add_documents(docs)
    vectorstore.persist()

    print("Ingestion complete!")


def ask(query):
    retriever = vectorstore.as_retriever()

    # Retrieval step
    retrieved_docs = retriever.invoke(query)

    # RAG Prompt
    template = """
    You are a real estate assistant. Use ONLY the following context to answer:

    {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate.from_template(template)

    # Build the chain using LCEL
    chain = (
        {
            "context": lambda x: "\n\n".join([d.page_content for d in retrieved_docs]),
            "question": lambda x: query,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({})


# Example usage
if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    ingest(urls)
    print(ask("What was the 30-year fixed mortgage rate and date?"))
