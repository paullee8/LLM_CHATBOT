from langchain_openai.chat_models import AzureChatOpenAI

def model_seletion(openai_api_base_value, openai_api_version_value,deployment_name_value, openai_api_key_value, openai_api_type_value,temperature_level):
    llm_model = AzureChatOpenAI(azure_endpoint=openai_api_base_value, 
                    openai_api_version=openai_api_version_value, deployment_name=deployment_name_value, 
                    openai_api_key=openai_api_key_value, openai_api_type = openai_api_type_value,
                    temperature=temperature_level) 
    
    return llm_model, temperature_level