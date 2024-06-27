from langchain_community.chat_models.friendli import ChatFriendli
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.openai import ChatOpenAI
from anthropic import Anthropic

from tokencost import calculate_prompt_cost, calculate_completion_cost, count_string_tokens

from config import config


def calculate_tokens(prompt: str, result: str, model: str) -> tuple[float, float, int, int]:
    """정해진 모델에 따른 프롬프트와 완성에 따른 비용 계산 함수. 

    Args:
        prompt (str): 추론에 사용된 프롬프트. 
        completion (str): 프롬프트 추론 후 나온 결과. 
        model (str): 추론시 사용된 모델명. 

    Returns:
        tuple[float, float, int, int]: 프롬프트 가격, 결과 가격, 프롬프트 토큰수, 결과 토큰수
    """

    cost_prompt = calculate_prompt_cost(prompt, model)
    cost_response = calculate_completion_cost(result, model)
    tokens_prompt = count_string_tokens(prompt, model)
    tokens_result = count_string_tokens(result, model)

    print(f">>> Prompt Cost: ${cost_prompt} Tokens: {tokens_prompt}")
    print(f">>> completion Cost: ${cost_response} Tokens: {tokens_result}")

    return (cost_prompt, cost_response, tokens_prompt, tokens_result)


def request_to_claude(prompt: str) -> str:
    """Anthropic의 claude 모델에게 prompt 추론 및 결과를 반환하는 함수. 

    Args:
        prompt (str): 모델에 추론시킬 프롬프트. 

    Returns:
        str: 프롬프트 추론 후 결과. 
    """

    model = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = model.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    result = message.content[0].text

    return result


def request_to_friendli(prompt: str) -> str:
    """Friendli에서 서비스중인 모델에게 prompt 추론 및 결과를 반환하는 함수. 

    Args:
        prompt (str): 모델에 추론시킬 프롬프트. 

    Returns:
        str: 프롬프트 추론 후 결과. 
    """

    model = ChatFriendli(
            model="meta-llama-3-70b-instruct", friendli_token=config.FRIENDLI_TOKEN
    )
    result = model.invoke(prompt).content

    return result


def request_to_ollama(prompt: str) -> str:
    """Ollama에서 서비스중인 모델에게 prompt 추론 및 결과를 반환하는 함수. 

    Args:
        prompt (str): 모델에 추론시킬 프롬프트. 

    Returns:
        str: 프롬프트 추론 후 결과. 
    """

    model = ChatOllama(model="llama3:70b", base_url="http://localhost:7869")
    result = model.invoke(prompt).content

    return result


def request_to_gpt3p5(prompt: str) -> str:
    """OpenAI에서 GPT-3.5에게 prompt 추론 및 결과를 반환하는 함수. 

    Args:
        prompt (str): 모델에 추론시킬 프롬프트. 

    Returns:
        str: 프롬프트 추론 후 결과. 
    """

    model = ChatOpenAI(
        temperature=0.7,
        max_tokens=2048,
        model="gpt-3.5-turbo",
        api_key=config.OPENAI_GPT_KEY,
    )
    result = model.invoke(prompt).content

    return result


def request_to_llm(prompt: str):
    return request_to_gpt35(prompt)
