from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

class retrieve_docu_qa():
    def __init__(self, llm_model, lang_mode,file_path, user_input, HuggingFaceEmbeddings_model_name):
        self.llm_model = llm_model
        self.lang_mode = lang_mode
        self.file_path = file_path
        self.user_input = user_input
        self.embeddings = HuggingFaceEmbeddings(model_name=HuggingFaceEmbeddings_model_name)

    def embed_pdf(self):
        self.loader = PyPDFLoader(self.file_path)
        self.pages = self.loader.load_and_split()
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.db = Chroma.from_documents(self.pages, self.embeddings)
        self.retriever = self.db.as_retriever()
        return self.retriever
    
    def embed_txt(self):
        self.loader = TextLoader(self.file_path)
        self.txt_docu = self.loader.load_and_split()
        self.db = Chroma.from_documents(self.txt_docu, self.embeddings)
        self.retriever = self.db.as_retriever()
        return self.retriever
    
    def retrieve_query(self):
        if self.file_path.endswith("pdf"):
            self.embed_db = self.embed_pdf()
        elif self.file_path.endswith("txt"):
            self.embed_db = self.embed_txt()

        self.template = """Provide a very detailed answer to the question based only on the following context:
        {context}
        Question: {user_question}

        Provide your very detailed answer in {lang}.
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.retrieval_pdf_txt_chain = (
            {
                "context": itemgetter("user_question") | self.embed_db,
                "user_question": itemgetter("user_question"),
                "lang": itemgetter("lang")
            }
            | self.prompt
            | self.llm_model
            | StrOutputParser()
        )
        self.llm_answer = self.retrieval_pdf_txt_chain.invoke({'user_question':self.user_input, 'lang':self.lang_mode})
        return self.llm_answer
