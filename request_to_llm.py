from langchain_community.chat_models.friendli import ChatFriendli
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.openai import ChatOpenAI
from anthropic import Anthropic

from config import config

# import instructor


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
    return model.invoke(prompt).content

def request_to_llm(prompt: str):
    return request_to_gpt35(prompt)

if __name__ == "__main__":
    with open("sample_prompt.txt", "r") as file:
        print(request_to_llm(file.read()))
