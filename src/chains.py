import os
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.callbacks.base import BaseCallbackHandler
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, intial_text=""):
        self.container = container
        self.text = intial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        # self.container.markdown(self.text)
        # print(self.text)


class LLMChainConfig:
    def __init__(self, config={}) -> None:
        self._config = config
        self._embedding = self._load_embeddings()
        self._llm = self._load_llm()
        self._configure_qa_rag_chain()

    def _load_embeddings(self):
        return SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2", cache_folder=self._config["cache_folder"]
        )

    def _load_llm(self):
        return ChatOllama(
            temperature=0,
            base_url=self._config["ollama_base_url"],
            model="mistral",
            streaming=True,
            top_k=10,
            top_p=0.3,
            num_ctx=3072,
        )

    def _configure_qa_rag_chain(self):
        system_template = """
        Use the following pieces of context to create questions. The question can be 
        Multiple choice question, short answer question or long answer question.
        Make sure to rely on information from the answers and not on questions to provide accuate responses.
        Also make sure to provide the answers for the question.
        The format of the question should be JSON.
        Don't repeat questions and make sure the questions are relevant to the context.
        
        Context:
        {context}
        """

        self._qa_prompt = PromptTemplate(
            template=system_template, input_variables=["context", "question"]
        )

    def run(self, pdf):
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, length_function=len
        )

        chunks = text_splitter.split_text(text)

        v_store = Neo4jVector.from_texts(
            chunks,
            url=self._config["url"],
            username=self._config["username"],
            password=self._config["password"],
            embedding=self._embedding,
            index_name="quiz_ai",
            node_label="PdfDataChunk",
            pre_delete_collection=True,
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        return ConversationalRetrievalChain.from_llm(
            self._llm,
            v_store.as_retriever(),
            chain_type="stuff",
            memory=memory,
            combine_docs_chain_kwargs={"prompt": self._qa_prompt},
        )
