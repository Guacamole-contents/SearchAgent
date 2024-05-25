import argparse

from datetime import datetime

import json

from getmeta import makevideopair
from generate_prompt import generate_prompt
from save_result_to_db import save_result_to_db
from request_to_llm import request_to_llm


def main():
    # 입력 인자값 설정.
    parser = argparse.ArgumentParser(
        description='YouTube 영상의 코드를 이용하여 위반 영상을 찾아 DB에 저장합니다. ')
    parser.add_argument('--code', metavar='c', type=str,
                        help='YouTube의 영상코드. (Example: UP2RFQCszdk)')
    args = parser.parse_args()

    video_code = args.code

    list_data_result = makevideopair(video_code)

    for data_results in list_data_result:
        for data in data_results:
            origin_meta = data["origin_meta"]
            origin_comment = data["origin_comment"]
            violate_meta = data["violate_meta"]
            violate_comment = data["violate_comment"]

            prompt = generate_prompt(
                origin_meta, origin_comment, violate_meta, violate_comment)
            print(prompt)

            result_llm = request_to_llm(prompt)
            json_result = json.loads(result_llm)

            print("json_result:", json_result)
            print({"origin_video_name": video_code, "copy_video_name": violate_meta["Original video code"],
                   "is_copy": json_result["is_copy"], "reason": json_result["reason"]})

            save_result_to_db(
                [{"origin_video_name": video_code, "copy_video_name": violate_meta["Original video code"],
                 "is_copy": json_result["is_copy"], "reason": json_result["reason"], "search_date": int(datetime.now().timestamp())}])


if __name__ == '__main__':
    main()
