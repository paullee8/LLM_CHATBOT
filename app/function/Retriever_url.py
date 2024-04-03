from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (ChatPromptTemplate,PromptTemplate)
from bs4 import BeautifulSoup as Soup
from operator import itemgetter
import requests

class retrieve_url_qa():
    def __init__(self, llm_model ,lang_mode,bot_rule, url_input, user_input,HuggingFaceEmbeddings_model_name):
        self.llm_model = llm_model
        self.lang_mode = lang_mode
        self.bot_rule = bot_rule
        self.url_input = url_input
        self.user_input = user_input
        self.embeddings = HuggingFaceEmbeddings(model_name=HuggingFaceEmbeddings_model_name)

    def retriever_url(self):

        self.response = requests.get(self.url_input)
        if self.response.status_code != 200:
            return "invalid URL \nError message:" + str(self.response.status_code)
        
        self.loader = WebBaseLoader(
            self.url_input
        )
        self.url_docu = self.loader.load()
        self.db = Chroma.from_documents(self.url_docu, self.embeddings)
        self.url_retriever = self.db.as_retriever()

        self.template = """
                Provide a very detailed answer to the question based only on the following context:
                {context}
                Question: {user_question}
                
                Provide your very detailed answer in {lang}.
                """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.url_retrieval_chain = (
            {
                "context": itemgetter("user_question") | self.url_retriever,
                "user_question": itemgetter("user_question"),
                "lang": itemgetter("lang")
            }
            | self.prompt
            | self.llm_model
            | StrOutputParser()
        )
        self.llm_answer = self.url_retrieval_chain.invoke({'user_question':self.user_input, 'lang':self.lang_mode})
        return self.llm_answer