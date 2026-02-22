from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.youtube_transcript import extract_video_id, fetch_transcript

app = FastAPI()

# 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 以后可以改成指定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- 请求模型 --------
class VideoRequest(BaseModel):
    url: str


# -------- 路由 --------
@app.post("/api/transcript")
def get_transcript(request: VideoRequest):
    try:
        video_id = extract_video_id(request.url)
        data = fetch_transcript(video_id)
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)