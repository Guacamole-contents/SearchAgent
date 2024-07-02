from argparse import Namespace
import argparse

from pprint import pprint
from typing import List, Union, Annotated

from datetime import datetime, timedelta, timezone

import json

import jwt
from pydantic import BaseModel

from config import config, Config
from getmeta import make_video_pair
from generate_prompt import generate_violation_detection_prompt
from save_result_to_db import save_result_to_db
from request_to_llm import request_to_llm, validate_model_provider

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm


class Token(BaseModel):
    access_token: str
    token_type: str


def __run_model(args: Namespace, config: Config) -> List:
    """실행시 입력값과 설정 파일 정의에 따른 전체 결과를 반환하는 함수.

    Args:
        args (Namespace): 터미널 입력시 참조할 변수들.
        config (Config): 설정 파일에 정의된 변수들.

    Returns:
        List: 최종 결과의 리스트.
    """

    list_result = []
    video_code = args.code

    list_data_result = make_video_pair(
        video_code, config.MODEL_KEYWORD_EXTRACTION, config.PROVIDER_KEYWORD_EXTRACTION
    )

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
            result_llm = request_to_llm(
                prompt,
                config.MODEL_VIOLATION_DETECTION,
                config.PROVIDER_VIOLATION_DETECTION,
            )
            print(">>> Result of LLM")
            result_preprocessed = "{" + result_llm.split("{")[1].split("}")[0] + "}"
            print(result_preprocessed)
            json_result = json.loads(result_preprocessed)

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

            list_result.append(result_preprocessed)
            # TODO: 추후 DB insertion 작업 필요.
            # save_result_to_db(
            #     [{"origin_video_name": video_code, "copy_video_name": violate_meta["Original video code"],
            #      "is_copy": json_result["is_copy"], "reason": json_result["reason"], "search_date": int(datetime.now().timestamp())}])

    return list_result


def __check_config():
    """설정 파일을 검증하는 함수."""
    print(">>> Config arguments")
    pprint(config)

    # config.py의 모델 설정 검증.
    is_match = validate_model_provider(
        config.MODEL_KEYWORD_EXTRACTION, config.PROVIDER_KEYWORD_EXTRACTION
    )
    if not is_match:
        print("!!! Model provider mismatch.")
        exit(1)

    is_match = validate_model_provider(
        config.MODEL_VIOLATION_DETECTION, config.PROVIDER_VIOLATION_DETECTION
    )
    if not is_match:
        print("!!! Model provider mismatch.")
        exit(1)


def create_access_token(data: dict, expires_delta: timedelta) -> str:
    """입력받은 데이터와 만료 날짜를 이용하여 JWT 토큰을 발행하는 함수.

    Args:
        data (dict): JWT 토큰 발행 시 들어갈 데이터.
        expires_delta (timedelta): 만료 기한.

    Returns:
        str: 생성된 JWT.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM
    )

    return encoded_jwt


# 설정 유효성 확인 후 HTTP 서버 실행.
__check_config()
app = FastAPI()


@app.get("/v1/worker/")
def run(v: Union[str, None]):
    # TODO: 인자 v를 받아 실제 동작 처리.
    #
    # 동작 실행 전, worker 사용 가능 여부 확인.
    # 모든 worker가 소진된 경우 사용 불가 에러 처리.
    # 실행에 따른 worker 관리가 되어야함.
    #
    # 해당 기능은 JWT 토큰 발행 여부 확인 필요.
    #
    # 추론시 함수 `__run_model` 사용.
    return {"v": v}


@app.get("/v1/worker/status")
def health():
    # TODO: 현재 사용 가능한 worker 갯수와 사용 가능 여부 통보.
    # 해당 기능은 JWT 토큰 확인 필요 없음.
    return {"is_avaiable": True, "available_workers": 5}


@app.post("/v1/agent/auth")
def auth(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token:
    """사용자 인증을 위한 API 함수.

    Args:
        form_data (Annotated[OAuth2PasswordRequestForm, Depends): 사용자 요청 데이터.

    Raises:
        HTTPException: 로그인 실패시 발생.

    Returns:
        Token: 로그인 성공시 발행된 JWT 토큰.
    """

    # 입력된 사용자 정보.
    passwd = form_data.password
    agent_name = form_data.username

    # 입력된 정보 유효성 확인.
    if passwd != config.AGENT_PASSWORD or agent_name != config.AGENT_NAME:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect Agent ID or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 토큰 발행.
    date_access_token = timedelta(minutes=config.JWT_EXPRIE_MINUTES)
    access_token = create_access_token(
        data={"sub": config.AGENT_NAME}, expires_delta=date_access_token
    )

    return Token(access_token=access_token, token_type="bearer")
