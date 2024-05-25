import jinja2


def generate_prompt(origin_data_metadata: dict, origin_data_comments: list, copy_data_metadata: dict, copy_data_comments: list, path_template: str = "./prompt_template.txt") -> str:
    """주어진 원본 영상 데이터와 위반 후보 영상 데이터를 이용하여 위반 검출 프롬프트를 제공하는 함수. 

    Args:
        origin_data_metadata (dict): 원본 영상의 YouTube 메타데이터. 
        origin_data_comments (list): 원본 영상의 YouTube 댓글 데이터. 
        copy_data_metadata (dict): 위반 영상의 YouTube 메타데이터. 
        copy_data_comments (list): 위반 영상의 YouTube 댓글 데이터. 
        path_template (str, optional): 프롬프트 템플릿 파일 경로. Defaults to "./prompt_template.txt".

    Returns:
        str: 프롬프트 결과. 
    """

    # Jinja template 사용을 위한 설정 및 템플릿 불러오기.
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template(path_template)

    # 원본 영상 데이터 추출.
    print(origin_data_metadata)
    print()
    print(origin_data_comments)
    print()
    print(copy_data_metadata)
    print()
    print(copy_data_comments)

    original_video_title = origin_data_metadata["Original title"]
    original_video_descriptoin = origin_data_metadata["Original body text"]
    original_comments = [x["text"] for x in origin_data_comments]

    # 위반 후보 영상 데이터 추출.
    copy_video_title = copy_data_metadata["Original title"]
    copy_video_descriptoin = copy_data_metadata["Original body text"]
    copy_comments = [x["text"] for x in copy_data_comments]

    # 프롬프트 템플릿에 데이터 적용.
    prompt = template.render(original_video_title=original_video_title,
                             original_video_descriptoin=original_video_descriptoin,
                             original_comments=original_comments,
                             copy_video_title=copy_video_title,
                             copy_video_descriptoin=copy_video_descriptoin,
                             copy_comments=copy_comments)

    return prompt
