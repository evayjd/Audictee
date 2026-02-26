from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, Optional
from app.youtube_transcript import extract_video_id, fetch_transcript
from app.transcript_whisper import transcribe_with_whisper
app = FastAPI()

# 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 以后可以改成指定域名(希望)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- 请求模型 --------
class VideoRequest(BaseModel):
    url: str
    engie: Literal["youtube", "whisper"] = "youtube"
    language: Optional[str] = "fr"
    model_size: str = "small"


# -------- 路由 --------
@app.post("/api/transcript")
def get_transcript(request: VideoRequest):
    try:
        if request.engie == "whisper":
            return transcribe_with_whisper(
                request.url,
                language=request.language,
                model_size=request.model_size,
            )
        video_id=extract_video_id(request.url)
        return fetch_transcript(video_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)