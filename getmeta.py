
import argparse
import json
import os
import urllib.parse
from pytube import YouTube
from googleapiclient.discovery import build
import anthropic
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from googleapiclient.errors import HttpError
from tqdm import tqdm

import time

load_dotenv()
api_key = os.getenv('anthropic_API_KEY')

def query_anthropic(input_content):
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": input_content}]
    )
    return message.content[0].text


prompt = """귀하는 GPT-4 아키텍처를 기반으로 OpenAI에 의해 학습된 대규모 언어 모델인 ChatGPT입니다.

귀하의 임무는 당사의 동영상을 침해하는 동영상을 찾기 위한 검색어를 생성하는 것입니다.

검색어를 생성하는 방법은 다음과 같습니다. 

- 저희가 제공하는 메타데이터는 7개의 키가 있는 JSON 형식이며, 각 키에는 다음 값이 포함되어 있습니다. 

1. '원래 크리에이터 이름': 원본 동영상을 게시한 채널의 이름입니다. 

2. '원본 제목': 원본 동영상의 제목입니다.

3. '원본 본문 텍스트': 원본 동영상과 함께 게시된 텍스트입니다. 콘텐츠일 수도 있지만 채널에 대한 공지 사항일 수도 있습니다. 

4. '원본 구독자 수': 오리지널 동영상 채널의 구독자 수입니다. 

5. '원본 조회수': 원본 동영상을 시청한 사람의 수입니다. 

6. '원래 업로드 날짜': 원본 동영상이 업로드된 날짜입니다. 

- 총 5개의 검색어를 생성해야 합니다. 

생성하는 방법을 설명하겠습니다. 

1) 우선, '원본 제목'과 '원본 본문'이 겹치는 경우 검색어에 포함시키는 것이 중요합니다. '원본 본문 텍스트'가 겹치는 경우에도 검색어에 포함시키는 것이 중요합니다. '원본 본문 텍스트'와 '원본 작성자 이름'이 겹치는 경우에도 문자를 확인할 수 있으므로 검색어에 포함시키는 것이 중요합니다. 

2) 겹치지 않는 경우 '원본 제목'이 가장 중요하고, '원본 크리에이터 이름'이 그 다음, '원본 본문'을 요약하는 단어가 그 다음으로 중요합니다 다섯 가지 검색어는 서로 다르지만 이 오리지널 동영상으로 만들 수 있는 쇼트 동영상을 검색하는 데 필요합니다. 

"가장 중요한 것은 한국어 동영상에 초점을 맞추기 때문에 검색어가 한국어로 되어 있어야 한다는 점입니다.

3) 중요한 것은 너가 검색어를 json 형태로 내보내야한다는 거야. 예시를 들면 
{
    "1" : "검색어 1 내용",
    "2" : "검색어 2 내용",
    "3" : "검색어 3 내용",
    "4" : "검색어 4 내용",
    "5" : "검색어 5 내용"
}
"""

def parse_arguments():
    """입력 인자를 파싱하는 함수.

    Returns:
        Namespace: argparse로부터 반환된 인자값.
    """
    parser = argparse.ArgumentParser(
        description='YouTube 영상의 댓글을 이용해, 영상 내 대사를 군집화하여 프롬프트로 반환해줍니다.')
    parser.add_argument('--code', metavar='c', type=str,
                        help='YouTube의 영상코드. (Example: UP2RFQCszdk)')
    return parser.parse_args(args=["--code", "pnRTzupkJbc"])


#     return comments
def get_video_comments(api_key, video_id: str) -> list:
    """YouTube 영상 ID를 이용하여 해당 영상의 댓글 목록을 가져오는 함수.
    likeCount 기준으로 내림차순 정렬 후 상위 4개 댓글을 반환합니다.

    Args:
        api_key (str): 사용자가 발급받은 API 키.
        video_id (str): YouTube 영상 코드.

    Returns:
        list: 지정 영상의 상위 4개 댓글 목록.
    """

    comments = []
    api_obj = build('youtube', 'v3', developerKey=api_key)
    response = api_obj.commentThreads().list(
        part='snippet,replies', videoId=video_id, maxResults=100).execute()

    while response:
        for item in tqdm(response['items']):
            try:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': comment.get('textDisplay', ''),
                    'author': comment.get('authorDisplayName', ''),
                    'publishedAt': comment.get('publishedAt', ''),
                    'likeCount': comment.get('likeCount', 0)
                })

                if item['snippet']['totalReplyCount'] > 0:
                    for reply_item in item['replies']['comments']:
                        reply = reply_item['snippet']
                        comments.append({
                            'text': reply.get('textDisplay', ''),
                            'author': reply.get('authorDisplayName', ''),
                            'publishedAt': reply.get('publishedAt', ''),
                            'likeCount': reply.get('likeCount', 0)
                        })
            except KeyError:
                continue

        if 'nextPageToken' in response:
            response = api_obj.commentThreads().list(
                part='snippet,replies', videoId=video_id,
                pageToken=response['nextPageToken'], maxResults=100).execute()
        else:
            break

    # 댓글을 likeCount 기준으로 내림차순 정렬하고 상위 4개만 반환합니다.
    top_comments = sorted(
        comments, key=lambda x: x['likeCount'], reverse=True)[:4]
    return top_comments

def get_video_title(api_key: str, video_id: str) -> str:
    """지정된 YouTube 영상의 제목을 가져오는 함수. 

    Args:
        api_key (str): 사용자가 발급받은 API 키. 
        video_id (str): YouTube 영상 코드. 

    Returns:
        str: 지정된 YouTube 영상의 제목. 
    """

    return __get_video_info_in_snippet(api_key, video_id, "title")


def get_video_description(api_key: str, video_id: str) -> str:
    """지정된 YouTube 영상의 요약 정보를 가져오는 함수. 

    Args:
        api_key (str): 사용자가 발급받은 API 키. 
        video_id (str): YouTube 영상 코드. 

    Returns:
        str: 지정된 YouTube 영상의 요약 정보
    """

    return __get_video_info_in_snippet(api_key, video_id, "description")


def get_channel_title(api_key: str, video_id: str) -> str:
    """지정된 YouTube 영상의 채널명을 가져오는 함수. 

    Args:
        api_key (str): 사용자가 발급받은 API 키. 
        video_id (str): YouTube 영상 코드. 

    Returns:
        str: 지정된 YouTube 영상의 채널명. 
    """

    return __get_video_info_in_snippet(api_key, video_id, "channelTitle")


def get_channel_id(api_key: str, video_id: str) -> str:
    """지정된 YouTube 영상 채널의 구독자 수를 가져오는 함수. 

    Args:
        video_id (str): 사용자가 발급받은 API 키. 
        channel_id (str): YouTube 영상의 채널 코드. 

    Returns:
        str: 지정된 YouTube 영상의 채널명. 
    """

    return __get_video_info_in_snippet(api_key, video_id, "channelId")


def get_channel_subscribers(api_key: str, channel_id: str) -> str:
    """지정된 YouTube 영상 채널의 구독자 수를 가져오는 함수. 

    Args:
        api_key (str): 사용자가 발급받은 API 키. 
        channel_id (str): YouTube 영상의 채널 코드. 

    Returns:
        str: 지정된 YouTube 영상의 채널명. 
    """

    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.channels().list(
        part='statistics',
        id=channel_id
    )
    response = request.execute()
    subscriber_count = int(response['items'][0]['statistics']['subscriberCount'])
    
    return subscriber_count


def get_video_hashtag(api_key: str, video_id: str) -> str:
    """지정된 YouTube 영상의 hashtag를 가져오는 함수. 

    Args:
        api_key (str): 사용자가 발급받은 API 키. 
        video_id (str): YouTube 영상 코드. 

    Returns:
        str: 지정된 YouTube 영상의 채널명. 
    """

    return __get_video_info_in_snippet(api_key, video_id, "tags")


def get_video_view_count(api_key: str, video_id: str) -> str:
    """지정된 YouTube 영상의 재생 횟수를 가져오는 함수. 

    Args:
        api_key (str): 사용자가 발급받은 API 키. 
        video_id (str): YouTube 영상 코드. 

    Returns:
        str: 지정된 YouTube 영상의 채널명. 
    """

    return __get_video_info_in_statistics(api_key, video_id, "viewCount")


def get_video_upload_date(api_key: str, video_id: str) -> str:
    """지정된 YouTube 영상의 업로드 일자를 가져오는 함수. (YYYY. MM. DD)

    Args:
        api_key (str): 사용자가 발급받은 API 키. 
        video_id (str): YouTube 영상 코드. 

    Returns:
        str: 지정된 YouTube 영상의 채널명. 
    """

    return __get_video_info_in_snippet(api_key, video_id, "publishedAt")


def __get_video_info_in_snippet(api_key: str, video_id: str, section: str) -> str:
    """지정된 YouTube 영상 정보 중 원하는 정보(section)를 가져오는 함수. (내부 라이브러리 용)

    Args:
        api_key (str): 사용자가 발급받은 API 키. 
        video_id (str): YouTube 영상 코드. 
        section (str): 요청할 정보명. 

    Returns:
        str: 요청한 정보 냐용. 
    """

    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videos().list(part='snippet,statistics,contentDetails', id=video_id)
    response = request.execute()

    information = response['items'][0]['snippet'][section]

    return information


def __get_video_info_in_statistics(api_key: str, video_id: str, section: str) -> str:
    """지정된 YouTube 영상 정보 중 원하는 정보(section)를 가져오는 함수. (내부 라이브러리 용)

    Args:
        api_key (str): 사용자가 발급받은 API 키. 
        video_id (str): YouTube 영상 코드. 
        section (str): 요청할 정보명. 

    Returns:
        str: 요청한 정보 냐용. 
    """

    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videos().list(part='snippet,statistics,contentDetails', id=video_id)
    response = request.execute()

    information = response['items'][0]['statistics'][section]

    return information

def getmata(video_code):
    # YouTube API 접속을 위한 API Key 환경변수 가져오기.
    youtube_api_key = os.getenv("YT_KEY")
    if youtube_api_key == None:
        print("YouTube API 사용을 위한 키를 환경변수 `YT_KEY`로 지정 후 다시 실행시켜 주세요. ")
        exit(1)

    video_producer = get_channel_title(youtube_api_key, video_code)
    video_name = get_video_title(youtube_api_key, video_code)
    video_body = get_video_description(youtube_api_key, video_code)

    channel_id = get_channel_id(youtube_api_key, video_code)
    video_subscribe = get_channel_subscribers(youtube_api_key, channel_id)
    
    video_view_cnt = get_video_view_count(youtube_api_key, video_code)
    video_upload_date = get_video_upload_date(youtube_api_key, video_code).replace("-", ".").split("T")[0]

    result = {
        "Original creator name": video_producer,
        "Original title": video_name,
        "Original body text": video_body,
        "Original subscribe": video_subscribe,
        "Original view": video_view_cnt,
        "Original upload date": video_upload_date,
        "Original video code": video_code
    }
    print("result: ", result)
    return result

def getmataandcomments(video_code):
    # YouTube API 접속을 위한 API Key 환경변수 가져오기.
    youtube_api_key = os.getenv("YT_KEY")
    if youtube_api_key == None:
        print("YouTube API 사용을 위한 키를 환경변수 `YT_KEY`로 지정 후 다시 실행시켜 주세요. ")
        exit(1)


    video_producer = get_channel_title(youtube_api_key, video_code)
    video_name = get_video_title(youtube_api_key, video_code)
    video_body = get_video_description(youtube_api_key, video_code)
    video_comment = get_video_comments(youtube_api_key, video_code)
    
    channel_id = get_channel_id(youtube_api_key, video_code)
    video_subscribe = get_channel_subscribers(youtube_api_key, channel_id)
    
    video_view_cnt = get_video_view_count(youtube_api_key, video_code)
    video_upload_date = get_video_upload_date(youtube_api_key, video_code).replace("-", ".").split("T")[0]

    result = {
        "Original creator name": video_producer,
        "Original title": video_name,
        "Original body text": video_body,
        "Original subscribe": video_subscribe,
        "Original view": video_view_cnt,
        "Original upload date": video_upload_date
    }
    comment = video_comment
    new_result = {"result": result, "comment": comment}
    return new_result
def getcomments(video_code):
    # YouTube API 접속을 위한 API Key 환경변수 가져오기.
    youtube_api_key = os.getenv("YT_KEY")
    if youtube_api_key == None:
        print("YouTube API 사용을 위한 키를 환경변수 `YT_KEY`로 지정 후 다시 실행시켜 주세요. ")
        exit(1)
    video_comment = get_video_comments(youtube_api_key, video_code)
    
    return video_comment


def claude(input_file):
    response = query_anthropic(input_file)
    print("response",   response)
    return response

def generate_youtube_links(search_list):
    linklist = []
    for query in search_list:
        query+= " #shorts"
        encoded_query = urllib.parse.quote(query)
        youtube_url = f"https://www.youtube.com/results?search_query={encoded_query}\n"
        linklist.append(youtube_url)
    return linklist


def fetch_top_video_codes_from_search_results(search_url):
    # Selenium WebDriver 설정
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    # 주어진 URL로 이동
    driver.get(search_url)
    
    # 로딩 대기
    time.sleep(5)  # 페이지 로드를 위해 5초 정도 대기합니다.
    
    # 동영상 링크 찾기
    video_elements = driver.find_elements(By.TAG_NAME, 'ytd-video-renderer')
    
    # 상위 5개 동영상 코드 추출
    video_codes = []
    for video in video_elements[:5]:  # 상위 5개 동영상에 대해 실행
        link_element = video.find_element(By.TAG_NAME, 'a')
        video_url = link_element.get_attribute('href')
        video_code = video_url.split('v=')[1]
        video_codes.append(video_code)
    
    driver.quit()  # 브라우저 종료
    
    return video_codes
def youtube_search(api_key, search_query):
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # search.list 메소드를 이용하여 YouTube 검색 수행
    search_response = youtube.search().list(
        q=search_query,        # 검색어
        part='id,snippet',     # 가져올 데이터
        maxResults=5          # 최대 결과 수
    ).execute()

    videos = []
    # 검색 결과에서 동영상 ID 및 제목 추출
    for search_result in search_response.get('items', []):
        if search_result['id']['kind'] == 'youtube#video':
            videos.append({
                'id': search_result['id']['videoId'],
                'title': search_result['snippet']['title']
            })

    return videos


def makevideopair(video_code):
    origin_meta_data = getmata(video_code)
    print("meta data 크롤링 완료")
    origin_comment = getcomments(video_code)
    print("comments 크롤링 완료")
    input_prompt = prompt + str(origin_meta_data)
    claude_response = claude(input_prompt)
    print("claude 검색어 생성 완료  ")
    response_data = json.loads(claude_response)
    result = []
    # , response_data["2"], response_data["3"], response_data["4"], response_data["5"]]
    generated_query_keywords = [response_data["1"]]
    generated_meta_comment_pair_list = []
    for i in range(len(generated_query_keywords)):
        output3 = youtube_search(os.getenv('YT_KEY'), generated_query_keywords[i])
        violate_meta_list = [getmata(str(output3[k]["id"])) for k in range(len(output3))]
        violate_comment_list = [getcomments(str(output3[k]["id"])) for k in range(len(output3))]
        print("len(violate_meta_list)",len(violate_meta_list))
        
        for j in range(len(violate_meta_list)):
            generated_meta_comment_pair_one = {"origin_meta": origin_meta_data, "origin_comment": origin_comment, "violate_meta": violate_meta_list[j], "violate_comment": violate_comment_list[j]}
            generated_meta_comment_pair_list.append(generated_meta_comment_pair_one)
        result.append(generated_meta_comment_pair_list)
    return result
