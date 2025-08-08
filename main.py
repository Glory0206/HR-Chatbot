from fastapi import FastAPI
from pydantic import BaseModel

class HRDocumentRequest(BaseModel):
    document_text: str

# FastAPI 앱 인스턴스 생성
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "안녕하세요 HR chatbot입니다."}