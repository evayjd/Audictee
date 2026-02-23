import re
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from app.nlp import analyze_sentences

TIMEOUT_SECONDS = 5


def extract_video_id(url: str) -> str:
    """
    从 YouTube URL 提取 video_id
    """
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)

    if not match:
        raise ValueError("URL YouTube invalide.")

    return match.group(1)


def _fetch_with_language_priority(video_id: str):
    """
    法语优先:
    1. 手动法语
    2. 自动法语
    3. 任意字幕
    """

    yt = YouTubeTranscriptApi()
    transcript_list = yt.list(video_id)

    # 手动法语
    try:
        transcript = transcript_list.find_transcript(["fr"])
        return transcript.fetch()
    except Exception:
        pass

    # 自动法语
    for transcript in transcript_list:
        if transcript.language_code == "fr" and transcript.is_generated:
            return transcript.fetch()

    # fallback
    try:
        transcript = next(iter(transcript_list))
        return transcript.fetch()
    except StopIteration:
        raise RuntimeError("Aucun sous-titre disponible.")


def clean_text(text: str) -> str:
    """
    清理字幕噪声（英文 + 法语）
    """

    # 删除音乐符号
    text = re.sub(r"♪.*?♪", "", text)

    # 删除常见噪声
    text = re.sub(
        r"\[(music|musique|applause|applaudissements|laughter|rire|cheering|acclamations|inaudible).*?\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 删除整行括号说明
    if re.fullmatch(r"\(.*?\)", text.strip()):
        return ""

    return text.strip()


def merge_segments(raw_segments: List[Dict]) -> List[Dict]:
    """
    将碎片字幕合并为完整句子
    """

    sentences = []
    buffer = ""
    start_time = None
    seg_end = None

    for seg in raw_segments:
        text = seg["text"]
        seg_start = seg["start"]
        seg_end = seg["start"] + seg["duration"]

        if not text:
            continue

        if start_time is None:
            start_time = seg_start

        buffer += " " + text

        if text.strip().endswith((".", "!", "?")):
            sentences.append(
                {
                    "text": buffer.strip(),
                    "start": start_time,
                    "end": seg_end,
                }
            )
            buffer = ""
            start_time = None

    # 处理最后残留
    if buffer and start_time is not None:
        sentences.append(
            {
                "text": buffer.strip(),
                "start": start_time,
                "end": seg_end,
            }
        )

    return sentences


def fetch_transcript(video_id: str) -> Dict:
    """
    获取字幕 → 清理 → 合并 → 返回结构化 JSON
    """

    try:
        # 超时控制
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch_with_language_priority, video_id)
            transcript = future.result(timeout=TIMEOUT_SECONDS)

        # 标准化
        raw_segments = [
            {
                "text": clean_text(item.text),
                "start": float(item.start),
                "duration": float(item.duration),
            }
            for item in transcript
        ]

        # 过滤空文本
        raw_segments = [seg for seg in raw_segments if seg["text"]]

        # 合并为句子
        sentences = merge_segments(raw_segments)
        
        #NLP分析
        sentences=analyze_sentences(sentences)

        # 返回结构
        return {
            "video_id": video_id,
            "language": "fr",
            "sentences": sentences,
        }

    except FuturesTimeout:
        raise RuntimeError(
            "La requête vers YouTube a dépassé le délai autorisé (timeout)."
        )

    except TranscriptsDisabled:
        raise RuntimeError("Les sous-titres sont désactivés pour cette vidéo.")

    except NoTranscriptFound:
        raise RuntimeError("Aucun sous-titre disponible pour cette vidéo.")

    except VideoUnavailable:
        raise RuntimeError("Vidéo indisponible.")

    except Exception as e:
        raise RuntimeError(f"Erreur inattendue : {str(e)}")