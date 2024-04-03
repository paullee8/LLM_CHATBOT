from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import langchain_experimental
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import pandas as pd

class pandas_agent_module():
    def __init__(self, llm_model,user_input,lang_mode, df_options, df_dict):
        self.llm_model = llm_model
        self.user_input = user_input
        self.lang_mode = lang_mode
        self.df_options = df_options
        self.df_dict = df_dict
        
    def pandas_agent_job(self):
        self.input_df_lt = []
        for df in self.df_options:
            self.input_df_lt.append(self.df_dict[df])
        self.agent = create_pandas_dataframe_agent(
            self.llm_model,
            self.input_df_lt,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

        self.ans = self.agent.invoke(self.user_input)
        return list(self.ans.values())[1]