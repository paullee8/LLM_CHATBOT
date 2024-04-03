from langchain.prompts import (ChatPromptTemplate,PromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,PipelinePromptTemplate)
from langchain.chains import LLMChain


def main_llm_result(llm_model,lang_mode,bot_charactor, prompt_mode,customized_prompt_input,user_input, history, df_dict=None):

    full_template = """{introduction}, {example}, {input_template}"""
    full_prompt = PromptTemplate.from_template(full_template)

    introduction_template = """You are chatbot specificly for {people}. Response the answer in {lang} and provide example."""
    introduction_prompt = PromptTemplate.from_template(introduction_template)

    example_template = """{customized_prompt_input}"""
    example_prompt = PromptTemplate.from_template(example_template)

    df_template = """If there is information in {df_prompt_input}, prepare to use there as database to answer."""
    df_prompt = PromptTemplate.from_template(df_template)

    start_template = """# Chat history 
                        {chat_history}

                        Human: {human_input}
                        Chatbot: 
                        """
    start_prompt = PromptTemplate.from_template(start_template)

    input_prompts = [("introduction", introduction_prompt),("example", example_prompt),("input_template", start_prompt)]

    pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

    llm_chain = LLMChain(
        llm=llm_model,
        prompt=pipeline_prompt,
        verbose=True,
    )

    llm_answer = llm_chain.predict(people=bot_charactor, lang=lang_mode ,human_input=user_input, 
                                    customized_prompt_input=customized_prompt_input,
                                    df_prompt_input = df_dict, chat_history=history
                                    ) 
    return llm_answer