# SearchAgent

## Start
config.py에 API_KEY 입력
```
class Config:
    FRIENDLI_TOKEN: str = "your_friendli_token"
    MONGODB_ATLAS_CLUSTER_URI: str = (
         "your_mongo_link"
    )
    DB_NAME: str = "team-13"
    COLLECTION_NAME: str = "guacamole"
    YOUTUBE_API_KEY: str = "your_youtube_api_key"
```

**프롬프트 양식은 문의해주세요**

```
pip install -r requirements.txt
python main.py --code pnRTzupkJbc (YOUTUBE_VIDEO_ID)
gradio search.py
```
