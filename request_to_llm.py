import os
import time
import pandas as pd

from langchain_community.chat_models.friendli import ChatFriendli
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.openai import ChatOpenAI
from anthropic import Anthropic


from tokencost import (
    calculate_prompt_cost,
    calculate_completion_cost,
    count_string_tokens,
)

from config import config


from typing import Tuple


def calculate_tokens(
    prompt: str, result: str, model: str
) -> Tuple[float, float, int, int]:
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
        "anthropic": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "meta": ["meta-llama-3-70b-instruct"],
        "friendli": ["llama2:70b"],
        "openai": [
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4-turbo",
            "gpt-4",
        ],
    }
    print(f"model: {model}, provider: {provider}")
    # 제공사가 없는 경우 False 반환.
    if provider not in dict_provider.keys():
        return False

    # 제공사 내에 모델이 없는 경우 False 반환.
    if model not in dict_provider[provider]:
        return False

    return True


def request_to_anthropic(model_name: str, prompt: str) -> str:
    """Anthropic의 claude 모델에게 prompt 추론 및 결과를 반환하는 함수.

    Args:
        prompt (str): 모델에 추론시킬 프롬프트.

    Returns:
        str: 프롬프트 추론 후 결과.
    """

    model = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = model.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    result = message.content[0].text

    return result


def request_to_friendli(model_name: str, prompt: str) -> str:
    """Friendli에서 서비스중인 모델에게 prompt 추론 및 결과를 반환하는 함수.

    Args:
        prompt (str): 모델에 추론시킬 프롬프트.

    Returns:
        str: 프롬프트 추론 후 결과.
    """

    model = ChatFriendli(model=model_name, friendli_token=config.FRIENDLI_TOKEN)
    result = model.invoke(prompt).content

    return result


def request_to_ollama(model_name: str, prompt: str) -> str:
    """Ollama에서 서비스중인 모델에게 prompt 추론 및 결과를 반환하는 함수.

    Args:
        prompt (str): 모델에 추론시킬 프롬프트.

    Returns:
        str: 프롬프트 추론 후 결과.
    """

    model = ChatOllama(model=model_name, base_url="http://localhost:7869")
    result = model.invoke(prompt).content

    return result


def request_to_openai(model_name: str, prompt: str) -> str:
    """OpenAI에서 GPT-3.5에게 prompt 추론 및 결과를 반환하는 함수.

    Args:
        prompt (str): 모델에 추론시킬 프롬프트.

    Returns:
        str: 프롬프트 추론 후 결과.
    """

    model = ChatOpenAI(
        temperature=0.7,
        max_tokens=2048,
        model=model_name,
        api_key=config.OPENAI_GPT_KEY,
    )
    result = model.invoke(prompt).content

    return result


def request_to_llm(
    prompt: str, model: str, provider: str, result_file_name: str = "cost_result.csv"
) -> str:
    """입력된 모델과 제공사에 따라 prompt를 추론하고 결과를 반환하는 함수.

    Args:
        prompt (str): 실행시키기 위한 프롬프트.
        model (str): 제공사에서 제공하는 모델명.
        provider (str): 모델이 속한 제공사. (anthropic, friendli, ollama, openai)
        result_file_name (str, optional): 비용 결과가 저장될 파일명. Defaults to "cost_result.csv".

    Raises:
        ValueError: 제공사와 모델이 맞지 않는 경우 발생.
        ValueError: 지원하지 않는 제공사인 경우 발생.

    Returns:
        str: 입력된 모델에 대해 프롬프트를 추론한 결과.
    """

    # 제공사와 모델간의 일치 여부 확인.
    if not validate_model_provider(model, provider):
        raise ValueError("Invalid model or provider")

    # 제공사와 모델에 따른 LLM 서비스 이용.
    if provider == "anthropic":
        result = request_to_anthropic(model, prompt)
    elif provider == "friendli":
        result = request_to_friendli(model, prompt)
    elif provider == "ollama":
        result = request_to_ollama(model, prompt)
    elif provider == "openai":
        result = request_to_openai(model, prompt)
    else:
        raise ValueError("Unsupported provider")

    # 모델과 프롬프트 및 결과에 따른 비용 계산 및 저장.
    result_tokens = calculate_tokens(prompt, result, model)

    # 결과 CSV으로 저장.
    dict_to_json = {
        "prompt": prompt,
        "result": result,
        "model": model,
        "provider": provider,
        "cost_prompt": result_tokens[0],
        "cost_result": result_tokens[1],
        "tokens_prompt": result_tokens[2],
        "tokens_result": result_tokens[3],
        "timestamp": time.strftime("%c", time.localtime()),
    }
    df_cost = pd.DataFrame(dict_to_json, index=[0])

    # 파일이 존재하는지 확인.
    if os.path.exists(result_file_name):
        # 파일이 존재하면 내용을 추가
        existing_df = pd.read_csv(result_file_name)
        updated_df = pd.concat([existing_df, df_cost], ignore_index=True)
    else:
        # 파일이 존재하지 않으면 새 파일을 생성.
        updated_df = df_cost

    # 결과 파일 저장.
    updated_df.to_csv(result_file_name, index=False)

    return result
