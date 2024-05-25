from dataclasses import dataclass


@dataclass
class Config:
    FRIENDLI_TOKEN: str = ""
    MONGODB_ATLAS_CLUSTER_URI: str = ""
    DB_NAME: str = ""
    COLLECTION_NAME: str = ""
    YOUTUBE_API_KEY: str = ""


config = Config()
