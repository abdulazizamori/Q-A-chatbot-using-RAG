# Q-A-using-RAG

LangChain Text Generation and Retrieval
This project demonstrates how to use LangChain to build a text generation and retrieval system using a combination of Hugging Face models, embeddings, and FAISS for efficient vector storage and retrieval.

Requirements
Before running the project, install the necessary libraries:

bash
Copy code
!pip install langchain
!pip install langchain-community
!pip install sentence-transformers
!pip install faiss-gpu
Code Explanation
Setting Up Hugging Face Pipeline:

We initialize a Hugging Face pipeline for text generation using the GPT-2 model.
We use LangChain to wrap the pipeline and integrate it into the overall retrieval system.
python
Copy code
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Initialize a Hugging Face pipeline
hf_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=200)

# Wrap it for LangChain
model = HuggingFacePipeline(pipeline=hf_pipeline)
Document Loader:

We load a CSV file containing FAQ data using CSVLoader and specify the encoding to handle potential character encoding issues.
python
Copy code
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='/content/codebasics_faqs.csv',
                   source_column="prompt",
                   encoding='latin1')  # Or try 'Windows-1252' if 'latin1' doesn't work

docs = loader.load()
Using Embeddings:

We use the EmbaasEmbeddings class from LangChain Community to embed queries, enabling efficient document retrieval based on similarity.
python
Copy code
from langchain_community.embeddings import EmbaasEmbeddings

embaas_api_key = "emb_1416341ad503f5f1da2bb63a9cdce0e458059c97b64fce3c"

embeddings = EmbaasEmbeddings(
    embaas_api_key=embaas_api_key,
    instruction="Represent the question for retrieval: "
)
VectorStore Setup:

FAISS is used as the vector store to store and search the embeddings. We create a FAISS instance from the documents and embeddings.
python
Copy code
from langchain.vectorstores import FAISS

vectordb = FAISS.from_documents(documents=docs,
                                embedding=embeddings)

retriever = vectordb.as_retriever(score_threshold=0.7)
Prompt Template for Answer Generation:

A custom prompt template is defined to generate answers based on the provided context and question, ensuring the answer is derived from the source document.
python
Copy code
from langchain.prompts import PromptTemplate

prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""
RetrievalQA Chain:

The RetrievalQA chain is used to combine the model, retriever, and prompt into a pipeline that retrieves relevant documents and generates answers based on those documents.
python
Copy code
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(llm=model,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    input_key="query",
                                    return_source_documents=True,
                                    chain_type_kwargs=chain_type_kwargs)
Example Usage
To get an answer to a question, you can query the system as follows:

python
Copy code
response = chain.run("What is your refund policy?")
print(response)
This will return an answer based on the relevant context found in the CSV document, or say "I don't know" if no relevant context is found.

Conclusion
This setup integrates various LangChain components, including text generation, document loading, embeddings, and retrieval, to create a powerful QA system. It leverages Hugging Face models, FAISS for vector storage, and custom prompt templates to ensure accurate answers.
