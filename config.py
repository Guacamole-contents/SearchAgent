from dataclasses import dataclass


@dataclass
class Config:
    FRIENDLI_TOKEN: str = ""
    MONGODB_ATLAS_CLUSTER_URI: str = ""
    DB_NAME: str = ""
    COLLECTION_NAME: str = ""
    YOUTUBE_API_KEY: list = [
        "<YOUR_GoogleAPIKey_HERE>",
        "<YOUR_GoogleAPIKey_HERE>",
        "<YOUR_GoogleAPIKey_HERE>",
    ]

    # Task 별 사용될 모델명과 제공사.
    MODEL_KEYWORD_EXTRACTION: str = "gpt-3.5-turbo"
    PROVIDER_KEYWORD_EXTRACTION: str = "openai"
    MODEL_VIOLATION_DETECTION: str = "gpt-3.5-turbo"
    PROVIDER_VIOLATION_DETECTION: str = "openai"

    # 지원하는 모델 및 제공사 목록.
    PROVIDER_MODEL_PAIR = {
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

config = Config()
