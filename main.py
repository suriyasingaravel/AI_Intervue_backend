import os
import uuid
import json
import re
from io import BytesIO
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from openai import OpenAI

# -------------------------------------------------------------------
# Load environment
# -------------------------------------------------------------------
load_dotenv()

# DO NOT raise errors at import time – Vercel imports this module to
# start the serverless function. If something raises here, you get a 500.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

parser = StrOutputParser()

# -------------------------------------------------------------------
# Helper: get LLM lazily (only when needed)
# -------------------------------------------------------------------
def get_llm():
    """
    Create ChatOpenAI instance only when needed.
    If OPENAI_API_KEY is missing, raise a FastAPI HTTPException
    instead of crashing the whole function at import time.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set in environment variables."
        )
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.7,
    )


# -------------------------------------------------------------------
# PROMPTS
# -------------------------------------------------------------------
GENERATE_QUESTIONS_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert technical interviewer. Based on the job role and experience level, generate exactly 5 relevant technical questions.

Job Role: {job_role}
Experience Level: {experience} years

Generate 5 questions that are:
1. Appropriate for the experience level
2. Technical and role-specific
3. Progressive in difficulty
4. Cover different aspects of the role

Return ONLY a JSON array of 5 questions, no explanations:
[
    "Question 1",
    "Question 2",
    "Question 3",
    "Question 4",
    "Question 5"
]
"""
)

EVALUATE_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert technical interviewer evaluating a candidate's answer.

Job Role: {job_role}
Experience Level: {experience} years
Question: {question}
Candidate's Answer: {answer}

Provide a detailed evaluation including:
1. Technical accuracy (1-10)
2. Completeness of answer (1-10)
3. Clarity of explanation (1-10)
4. Specific feedback and suggestions
5. Overall score (1-10)

Format your response as:
Score: X/10
Technical Accuracy: X/10
Completeness: X/10
Clarity: X/10
Feedback: [Your detailed feedback here]
"""
)

FINAL_REPORT_PROMPT = ChatPromptTemplate.from_template(
    """
You are an expert technical interviewer creating a comprehensive interview report.

Job Role: {job_role}
Experience Level: {experience} years

Interview Results:
{interview_results}

Create a professional final report including:
1. Overall assessment
2. Strengths identified
3. Areas for improvement
4. Technical competency score (average of all scores)
5. Recommendation (Pass/Fail/Consider with conditions)
6. Detailed breakdown of each question

Format as a professional report.
"""
)

# -------------------------------------------------------------------
# LLM helpers
# -------------------------------------------------------------------
def extract_json(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        json_text = match.group(0)
        try:
            return json.loads(json_text)
        except Exception:
            pass

    raise HTTPException(
        status_code=502,
        detail=f"LLM returned invalid JSON: {repr(raw)}"
    )


def generate_questions(job_role: str, experience: int) -> List[str]:
    llm = get_llm()
    chain = GENERATE_QUESTIONS_PROMPT | llm | parser
    raw = chain.invoke({"job_role": job_role, "experience": experience})
    if not raw or not raw.strip():
        raise HTTPException(
            status_code=502,
            detail="LLM did not return any data. Check API key and network.",
        )
    questions = extract_json(raw)
    questions = [str(q).strip() for q in questions]
    if len(questions) != 5:
        raise HTTPException(
            status_code=502,
            detail="LLM did not generate exactly 5 questions.",
        )
    return questions


def evaluate_answer(job_role: str, experience: int, question: str, answer: str) -> str:
    try:
        llm = get_llm()
        chain = EVALUATE_ANSWER_PROMPT | llm | parser
        return chain.invoke(
            {
                "job_role": job_role,
                "experience": experience,
                "question": question,
                "answer": answer,
            }
        )
    except Exception as e:
        print("Error in evaluate_answer:", e)
        raise HTTPException(status_code=500, detail="Evaluation failed")


def build_final_report(job_role: str, experience: int, data: List[dict]) -> str:
    interview_results = []
    for i, row in enumerate(data, start=1):
        interview_results.append(
            f"Question {i}: {row.get('question', '')}\n"
            f"Answer: {row.get('answer', '')}\n"
            f"Feedback: {row.get('feedback', '')}\n"
            + ("-" * 40)
        )
    results_blob = "\n".join(interview_results)

    llm = get_llm()
    chain = FINAL_REPORT_PROMPT | llm | parser
    final_report = chain.invoke(
        {
            "job_role": job_role,
            "experience": experience,
            "interview_results": results_blob,
        }
    )
    return final_report


# -------------------------------------------------------------------
# FastAPI setup
# -------------------------------------------------------------------
app = FastAPI(
    title="AI Interviewer API",
    description="",
    version="1.0.0",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    job_role: str = Field(..., example="React Developer")
    experience: int = Field(..., ge=0, le=50, example=2)


class SubmitAnswerRequest(BaseModel):
    answer: str


class CreateSessionResponse(BaseModel):
    session_id: str
    job_role: str
    experience: int
    questions: List[str]
    current_question_idx: int


class SubmitAnswerResponse(BaseModel):
    question_idx: int
    question: str
    feedback: str
    next_question_idx: Optional[int] = None
    next_question: Optional[str] = None


class SessionState(BaseModel):
    job_role: str
    experience: int
    data: List[Dict]
    current_question_idx: int
    final_report: Optional[str] = None


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice: str = Field(
        default="alloy",
        description="Voice to use (alloy, echo, fable, onyx, nova, shimmer)",
    )


# -------------------------------------------------------------------
# In-memory session store
# -------------------------------------------------------------------
sessions: Dict[str, SessionState] = {}


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to AI Interviewer"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
def create_session(payload: CreateSessionRequest):
    sid = uuid.uuid4().hex
    questions = generate_questions(payload.job_role, payload.experience)

    state = SessionState(
        job_role=payload.job_role,
        experience=payload.experience,
        data=[{"question": q, "answer": "", "feedback": ""} for q in questions],
        current_question_idx=0,
    )
    sessions[sid] = state

    return CreateSessionResponse(
        session_id=sid,
        job_role=state.job_role,
        experience=state.experience,
        questions=[q["question"] for q in state.data],
        current_question_idx=state.current_question_idx,
    )


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@app.post("/sessions/{session_id}/answers", response_model=SubmitAnswerResponse)
def submit_answer(session_id: str, payload: SubmitAnswerRequest):
    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    idx = state.current_question_idx
    if idx >= len(state.data):
        raise HTTPException(status_code=400, detail="All questions already answered")

    question = state.data[idx]["question"]

    state.data[idx]["answer"] = payload.answer.strip()

    feedback = evaluate_answer(
        state.job_role, state.experience, question, state.data[idx]["answer"]
    )
    state.data[idx]["feedback"] = feedback

    state.current_question_idx += 1

    next_q_idx = None
    next_q = None
    if state.current_question_idx < len(state.data):
        next_q_idx = state.current_question_idx
        next_q = state.data[next_q_idx]["question"]
    else:
        state.final_report = build_final_report(
            state.job_role, state.experience, state.data
        )

    sessions[session_id] = state

    return SubmitAnswerResponse(
        question_idx=idx,
        question=question,
        feedback=feedback,
        next_question_idx=next_q_idx,
        next_question=next_q,
    )


@app.get("/sessions/{session_id}/report")
def get_report(session_id: str):
    state = sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "job_role": state.job_role,
        "experience": state.experience,
        "final_report": state.final_report,
    }


# -------------------------------------------------------------------
# TTS
# -------------------------------------------------------------------
@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

        client = OpenAI(api_key=api_key)

        tts_response = client.audio.speech.create(
            model="tts-1",
            voice=request.voice,
            input=request.text,
        )

        audio_bytes = b"".join(tts_response.iter_bytes())

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"},
        )
    except HTTPException:
        raise
    except Exception as e:
        print("TTS Error:", e)
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")


# -------------------------------------------------------------------
# STT
# -------------------------------------------------------------------
@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")

        allowed_extensions = [
            ".mp3",
            ".mp4",
            ".mpeg",
            ".mpga",
            ".m4a",
            ".wav",
            ".webm",
        ]
        file_extension = os.path.splitext(audio.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported audio format: {file_extension}. "
                    f"Supported: {', '.join(allowed_extensions)}"
                ),
            )

        audio_data = await audio.read()
        if len(audio_data) == 0:
            raise HTTPException(
                status_code=400, detail="Uploaded audio file is empty"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

        client = OpenAI(api_key=api_key)

        audio_file = BytesIO(audio_data)
        audio_file.name = audio.filename

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
            response_format="verbose_json",
        )

        return {
            "text": transcript.text,
            "language": transcript.language,
            "duration": transcript.duration,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"STT Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"STT Error: {str(e)}")
    finally:
        await audio.close()

