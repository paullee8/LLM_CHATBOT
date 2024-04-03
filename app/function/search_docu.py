from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class search_docu():
    def __init__(self, llm_model, user_input, docu_list, docu_path_df):
        self.llm_model = llm_model
        self.user_input = user_input
        self.docu_list = docu_list
        self.docu_path_df = docu_path_df
    def find_docu_name(self):
        self.docu_template = """
                    Help me see if '{search_input}', contains similar word in list:{docu_list}.
                    Return your answer in following template:
                    Yes: return the word in the list
                    """

        self.docu_prompt = PromptTemplate.from_template(template=self.docu_template)

        self.llm_chain = (
            self.docu_prompt
            | self.llm_model
            | StrOutputParser()
        )
        
        self.llm_answer = self.llm_chain.invoke({'search_input':self.user_input, 'docu_list':self.docu_list})
        return(self.llm_answer)
    def calculate_similarity(self):
        self.docu_name = self.find_docu_name()
        self.results_with_scores = self.docu_path_df.similarity_search_with_score(self.docu_name) # a lower score is better https://python.langchain.com/docs/integrations/vectorstores/faiss
        for self.doc ,self.score in self.results_with_scores:
            if self.score<=1:
                self.answer = [self.doc.metadata,self.score]
                break
            else: self.answer = ['no',100]
        print(self.answer)
        return self.answer
