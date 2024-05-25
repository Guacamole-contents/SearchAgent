from pymongo import MongoClient

from datetime import datetime
from config import config
from typing import List, Dict


def _set_data_mapper(
    origin_video_name: str, copy_video_name: str, is_copy: bool, reason: str
):
    """
    mongoDB에 저장할 데이터의 형식을 생성
    :param origin_video_name: 원본 영상 이름
    :param copy_video_name: 침해 영상 이름
    :param is_copy: 침해쌍 여부
    :param reason: 침해 판단 사유

    :return: mongoDB에 저장할 데이터
    """
    return {
        "origin_video_name": origin_video_name,
        "copy_video_name": copy_video_name,
        "is_copy": is_copy,
        "reason": reason,
        "search_date": int(datetime.now().timestamp()),
    }


def save_result_to_db(data: List[Dict[str, str]]):
    """data를 저장하도록 요청한다.

    Args:
        data (_type_): DB에 origin video, copy video의 침해여부 결과 데이터


    Returns:
        result (InsertManyResult): collection 저장 결과 여부
    """

    with MongoClient(config.MONGODB_ATLAS_CLUSTER_URI) as client:
        # data = [
        #     {
        #         "origin_video_name": origin_video_name,
        #         "copy_video_name": copy_video_name,
        #         "is_copy": is_copy,
        #         "reason": reason,
        #         "search_date": int(datetime.now().timestamp()),
        #     }
        #     for origin_video_name, copy_video_name, is_copy, reason in data
        # ]

        print(data)

        return client[config.DB_NAME][config.COLLECTION_NAME].insert_many(data)


if __name__ == "__main__":
    ## Usage

    data = [
        ("origin1", "copy1", True, "reason1"),
        ("orisdgin2", "copy2", False, "reason2"),
        ("origin2", "copy2", False, "reason2"),
        ("origin2", "copy2", False, "reason2"),
        ("origin2132", "copy2", False, "reason2"),
        ("origindsf2", "copy2", False, "reason2"),
        ("origiasdn2", "copy2", False, "reason2"),
        ("origivbcn2", "copy2", False, "reason2"),
        ("asd", "copy2", False, "reason2"),
        ("origvcxivbcn2", "copy2", False, "reason2"),
        ("origdasivbcn2", "copy2", False, "reason2"),
        ("origadsivbcn2", "copy2", False, "reason2"),
        ("origiqwevbcn2", "copy2", False, "reason2"),
    ]

    result = save_result_to_db(data)
    print("done")
