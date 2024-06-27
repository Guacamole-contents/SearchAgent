import argparse

from datetime import datetime

import json

from config import config
from getmeta import makevideopair
from generate_prompt import generate_violation_detection_prompt
from save_result_to_db import save_result_to_db
from request_to_llm import request_to_llm, validate_model_provider


def main():
    # 입력 인자값 설정.
    parser = argparse.ArgumentParser(
        description="YouTube 영상의 코드를 이용하여 위반 영상을 찾아 DB에 저장합니다. "
    )
    parser.add_argument(
        "--code",
        metavar="c",
        type=str,
        help="YouTube의 영상코드. (Example: UP2RFQCszdk)",
    )
    args = parser.parse_args()

    video_code = args.code

    # config.py의 모델 설정 검증.
    is_match = validate_model_provider(config.MODEL_KEYWORD_EXTRACTION,
                                       config.PROVIDER_KEYWORD_EXTRACTION)
    if not is_match:
        print("!!! Model provider mismatch.")
        exit(1)

    is_match = validate_model_provider(config.MODEL_VIOLATION_DETECTION,
                                       config.PROVIDER_VIOLATION_DETECTION)
    if not is_match:
        print("!!! Model provider mismatch.")
        exit(1)

    # TODO: 해당 부분에서 설정된 모델에 따르도록 수정. (함수 makevideopair 내에서 함수 request_to_llm를 호출하는 부분. )
    list_data_result = makevideopair(video_code)

    for data_results in list_data_result:
        for data in data_results:
            origin_meta = data["origin_meta"]
            origin_comment = data["origin_comment"]
            violate_meta = data["violate_meta"]
            violate_comment = data["violate_comment"]

            prompt = generate_violation_detection_prompt(
                origin_meta, origin_comment, violate_meta, violate_comment
            )
            print(prompt)

            # TODO: 해당 부분에서 설정된 모델에 따르도록 수정.
            result_llm = request_to_llm(prompt)
            print(">>> Result of LLM")
            print(result_llm)
            json_result = json.loads(result_llm)

            print(">>> LLM Query result:", json_result)
            print(">>> Save result")
            print(
                {
                    "origin_video_name": video_code,
                    "copy_video_name": violate_meta["Original video code"],
                    "is_copy": json_result["is_copy"],
                    "reason": json_result["reason"],
                }
            )

            # save_result_to_db(
            #     [{"origin_video_name": video_code, "copy_video_name": violate_meta["Original video code"],
            #      "is_copy": json_result["is_copy"], "reason": json_result["reason"], "search_date": int(datetime.now().timestamp())}])


if __name__ == "__main__":
    main()
