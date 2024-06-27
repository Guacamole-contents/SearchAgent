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


config = Config()
