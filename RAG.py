import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS

#Setup flask server
app = Flask(__name__)
CORS(app)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Create a class to load relevant CSV columns only
class CSVColumnLoader:
    def __init__(self, file_path, columns):
        self.file_path = file_path
        self.columns = columns

    def load(self):
        # Load the CSV file and filter only the specified columns
        df = pd.read_csv(self.file_path, usecols=self.columns)
        # Combine all columns into a single text string per row
        docs = [
            Document(page_content=" ".join(row.dropna().astype(str)))
            for _, row in df.iterrows()
        ]
        return docs

columns = ["BrandName", "ModelGroup", "ModelNum", "Summed_Sales", "PartMFG", "PartNum", "Description", "LongDescription", "PartType"]
loader = CSVColumnLoader("random_sample.csv", columns)
docs = loader.load()

# Embed and store all the chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retriever that returns most similar chunks
retriever = vectorstore.as_retriever()

# Predefined prompt template
prompt = hub.pull("rlm/rag-prompt")

# Convert documents to strings
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Receive message from frontend and use it as the prompt

@app.route("/get-ai-message", methods=["POST"])
def get_ai_message():
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"message": "No query provided"}), 400

    response = ""
    for chunk in rag_chain.stream(user_query):
        response += chunk
    
    return jsonify({"role": "assistant", "content": response})

if __name__ == "__main__":
    app.run(port=5000)



vectorstore.delete_collection()

