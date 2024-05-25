from pymongo import MongoClient

import pandas as pd

import gradio as gr
from gradio_calendar import Calendar

from datetime import date, timedelta, datetime, timezone

client = MongoClient(
    "mongodb+srv://team-13:Zw3zHjTCqOdNvU2Q@cluster0.tyqdayd.mongodb.net/")
db = client["team-13"]
posts = db.posts


def convert_mongo_to_df(mongo_data) -> pd.DataFrame:
    """MongoDB의 결과를 DataFrame 형태로 변환하는 함수. 

    Args:
        mongo_data (_type_): MongoDB 색인 결과. 

    Returns:
        pd.DataFrame: 변환된 DataFrame. 
    """

    df = pd.DataFrame(mongo_data)

    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    df.columns = ["Original Title", "Copy Title",
                  "Is Copy", "Reason", "Search Date"]
    df['Search Date'] = df['Search Date'].apply(
        lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
    return df


def filter_records(title, records, title_original, is_op_copy, date_start: datetime, date_end: datetime) -> pd.DataFrame:
    """옵션을 이용해 데이터 필터링 후 DataFrame을 반환하는 함수. 

    Args:
        title (_type_): 사용하지 않음. 
        records (_type_): 기존 샘플 데이터셋. 
        title_original (_type_): 검색하기 위한 원본 영상 코드. 
        is_op_copy (bool): 위반 여부. 
        date_start (datetime): 검색 시작 날짜. 
        date_end (datetime): 검색 종료 날짜. 

    Returns:
        pd.DataFrame: 검색된 데이터셋. 
    """

    dict_search_op = {}

    if title_original != "":
        dict_search_op["origin_video_name"] = title_original
    if is_op_copy != "모두":
        is_copy = True if is_op_copy == "위반" else False
        dict_search_op["is_copy"] = is_copy
    dict_search_op["search_date"] = {"$gte": int(
        date_start.timestamp()), "$lte": int(date_end.timestamp())}

    data = convert_mongo_to_df(db.guacamole.find(dict_search_op))

    return data


# df_sample = pd.DataFrame(db.guacamole.find().limit(10))
df_sample = convert_mongo_to_df(db.guacamole.find().limit(10))

demo = gr.Interface(
    filter_records,
    [
        gr.Markdown("검색 옵션"),
        gr.Dataframe(
            value=df_sample,
            headers=["Original Title", "Copy Title",
                     "Is Copy", "Reason", "Search Date"],
            datatype=["str", "str", "bool", "str", "str"],
            col_count=(5, "fixed"),
        ),
        gr.Textbox(placeholder="YouTube"),
        gr.Radio(choices=["모두", "위반", "일반"], label="위반 여부",
                 value="모두", interactive=True),
        Calendar(type="datetime", label="검색 시작 날짜",
                 info="검색 포함 시작 날짜. ", value=str(date.today() - timedelta(days=30))),
        Calendar(type="datetime", label="검색 종료 날짜",
                 info="검색 포함 종료 날짜. ", value=str(date.today() + timedelta(days=1)))
    ],
    "dataframe",
    title="위반 검색 영상 결과",
    description="원본 영상 기준 키워드를 이용한 위반 영상 수집 결과. "
)

if __name__ == "__main__":
    demo.launch()
