from langchain_community.chat_models.friendli import ChatFriendli
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.openai import ChatOpenAI
from anthropic import Anthropic

from tokencost import calculate_prompt_cost, calculate_completion_cost

from config import config


def calculate_tokens(prompt: str, completion: str, model: str) -> tuple[int, int]:
    """정해진 모델에 따른 프롬프트와 완성에 따른 비용 계산 함수. 

    Args:
        prompt (str): 추론에 사용된 프롬프트. 
        completion (str): 프롬프트 추론 후 나온 결과. 
        model (str): 추론시 사용된 모델명. 

    Returns:
        tuple[int, int]: 각각 프롬프트의 추론 금액과 완성된 결과의 금액. 
    """
    prompt_string = "Hello world"
    response = "How may I assist you today?"

    prompt_cost = calculate_prompt_cost(prompt_string, model)
    completion_cost = calculate_completion_cost(response, model)
    print(f">>> Prompt Cost: ${prompt_cost}")
    print(f">>> completion Cost: ${completion_cost}")

    return (prompt_cost, completion_cost)


def validate_model_provider(model: str, provider: str) -> bool:
    """입력된 모델 제공사와 모델이 서로 맞는지 확인하는 함수. 

    Args:
        model (str): 사용하고자 하는 모델명. 
        provider (str): 사용하고자 하는 모델의 제공사.

    Returns:
        bool: 일치하면 True, 일치하지 않으면 False.
    """
    # 지원하는 모델 및 제공사 목록.
    dict_provider = {
        "anthropic": ["claude-3-opus-20240229"],
        "meta": ["meta-llama-3-70b-instruct"],
        "friendli": ["llama2:70b"],
        "gpt35": ["gpt-3.5-turbo"],
    }

    # 제공사가 없는 경우 False 반환.
    if provider in dict_provider[model]:
        return False

    # 제공사 내에 모델이 없는 경우 False 반환.
    if model in dict_provider[provider]:
        return False

    return True


def request_to_claude(prompt):
    model = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = model.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def request_to_friendli(prompt):
    model = ChatFriendli(
            model="meta-llama-3-70b-instruct", friendli_token=config.FRIENDLI_TOKEN
    )
    return model.invoke(prompt).content


def request_to_ollama(prompt):
    model = ChatOllama(model="llama3:70b", base_url="http://localhost:7869")
    return model.invoke(prompt).content


def request_to_gpt35(prompt):
    model = ChatOpenAI(
        temperature=0.7,
        max_tokens=2048,
        model="gpt-3.5-turbo",
        api_key=config.OPENAI_GPT_KEY,
    )

    result = model.invoke(prompt).content

    return model.invoke(prompt).content


def request_to_llm(prompt: str):
    return request_to_gpt35(prompt)
