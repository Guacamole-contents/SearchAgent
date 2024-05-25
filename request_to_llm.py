from langchain_community.chat_models.friendli import ChatFriendli
from config import config


def request_to_llm(prompt: str):
    model = ChatFriendli(
        model="meta-llama-3-70b-instruct", friendli_token=config.FRIENDLI_TOKEN
    )
    return model.invoke(prompt).content


if __name__ == "__main__":
    with open("sample_prompt.txt", "r") as file:
        print(request_to_llm(file.read()))
