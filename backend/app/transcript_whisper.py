# backend/app/transcript_whisper.py
import os
import re
import subprocess
import tempfile
from typing import List, Dict, Optional

from app.nlp import analyze_sentences
from app.youtube_transcript import extract_video_id  # 复用你已有的提取逻辑


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or f"Command failed: {' '.join(cmd)}")


def _download_audio(url: str, out_dir: str) -> str:

    out_tpl = os.path.join(out_dir, "audio.%(ext)s")
    _run(["yt-dlp", "-f", "bestaudio/best", "-o", out_tpl, url])

    files = [f for f in os.listdir(out_dir) if f.startswith("audio.")]
    if not files:
        raise RuntimeError("Download failed: No audio file generated (possible causes: permissions, geo-restriction, or network issues).")
    return os.path.join(out_dir, files[0])


def clean_text(text: str) -> str:
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
            sentences.append({"text": buffer.strip(), "start": start_time, "end": seg_end})
            buffer = ""
            start_time = None

    if buffer and start_time is not None:
        sentences.append({"text": buffer.strip(), "start": start_time, "end": seg_end})

    return sentences


def transcribe_with_whisper(
    url: str,
    language: Optional[str] = "fr",
    model_size: str = "small",
) -> Dict:

    try:
        # 提取 video_id
        video_id = extract_video_id(url)

        # 下载音频
        with tempfile.TemporaryDirectory() as td:
            audio_path = _download_audio(url, td)

            # Whisper 转写
            from faster_whisper import WhisperModel

            model = WhisperModel(model_size, device="auto", compute_type="auto")
            segments, info = model.transcribe(audio_path, language=language)

           
            raw_segments = []
            for s in segments:
                txt = clean_text(s.text)
                if not txt:
                    continue
                start = float(s.start)
                end = float(s.end)
                raw_segments.append(
                    {"text": txt, "start": start, "duration": max(0.0, end - start)}
                )

        # 合并为句子
        sentences = merge_segments(raw_segments)

        #  NLP 分析
        sentences = analyze_sentences(sentences)

        # language：优先用 whisper 识别结果（如果拿得到），否则用传入的 language
        detected_lang = getattr(info, "language", None) if "info" in locals() else None
        lang_out = detected_lang or language or "unknown"

        return {"video_id": video_id, "language": lang_out, "sentences": sentences}

    except Exception as e:
        raise RuntimeError(f"Whisper 转写失败：{str(e)}")