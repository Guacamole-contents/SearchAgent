import jinja2


def generate_violation_detection_prompt(
    origin_metadata: dict,
    origin_comments: list,
    copy_metadata: dict,
    copy_comments: list,
    template_path: str = "./prompt_template.txt",
) -> str:
    """
    주어진 원본 영상 데이터와 위반 후보 영상 데이터를 이용하여 위반 검출 프롬프트를 제공하는 함수.

    Args:
        origin_metadata (dict): 원본 영상의 YouTube 메타데이터.
        origin_comments (list): 원본 영상의 YouTube 댓글 데이터.
        copy_metadata (dict): 위반 영상의 YouTube 메타데이터.
        copy_comments (list): 위반 영상의 YouTube 댓글 데이터.
        template_path (str, optional): 프롬프트 템플릿 파일 경로. Defaults to "./prompt_template.txt".

    Returns:
        str: 프롬프트 결과.
    """
    # Jinja template 사용을 위한 설정 및 템플릿 불러오기
    template_loader = jinja2.FileSystemLoader(searchpath="./")
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(template_path)

    # 데이터 출력
    print(origin_metadata)
    print()
    print(origin_comments)
    print()
    print(copy_metadata)
    print()
    print(copy_comments)

    # 원본 영상 데이터 추출
    original_video_title = origin_metadata["Original title"]
    original_video_description = origin_metadata["Original body text"]
    original_comments = [comment["text"] for comment in origin_comments]

    # 위반 후보 영상 데이터 추출
    copy_video_title = copy_metadata["Original title"]
    copy_video_description = copy_metadata["Original body text"]
    copy_comments = [comment["text"] for comment in copy_comments]

    # 프롬프트 템플릿에 데이터 적용
    prompt = template.render(
        original_video_title=original_video_title,
        original_video_description=original_video_description,
        original_comments=original_comments,
        copy_video_title=copy_video_title,
        copy_video_description=copy_video_description,
        copy_comments=copy_comments,
    )

    return prompt
