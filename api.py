from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from handler import getOutput
from dotenv import load_dotenv
import os

app = FastAPI()

security = HTTPBasic()

load_dotenv()

USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(
        credentials.username,
        USERNAME
    )

    correct_password = secrets.compare_digest(
        credentials.password,
        PASSWORD
    )

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid Credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username




class QuestionRequest(BaseModel):
    question: str

@app.post("/execute")
def execute(
    request: QuestionRequest,
    user: str = Depends(authenticate)
):
    return getOutput(request.question)
