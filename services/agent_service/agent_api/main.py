import uuid
import jwt as pyjwt
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from shared.config import settings
from google.adk.runners import Runner
from google.genai import types
from agent.agent import root_agent
from agent.session import MongoSessionService
from pydantic import BaseModel, Field

security = HTTPBearer()



app = FastAPI(title="RiteCare Agent API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)
session_service = MongoSessionService()
runner = Runner(agent=root_agent, app_name="ritecare", session_service=session_service)

class QueryRequest(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    channel: str = "cli"
    user_id: str = "anonymous"
    bu_hint: str | None = None   # set by slack_gateway when channel maps to a known BU


class QueryResponse(BaseModel):
    response: str
    session_id: str


async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    try:
        payload = pyjwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        return {
            "user_id": payload["sub"],
            "role": payload["role"],
            "bu_access": payload.get("bu_access", [])
        }
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token Expired")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    user: dict = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    
    #Get or create session, then inject auth state
    session = await session_service.get_session(
        app_name="ritecare",
        user_id=user['user_id'],
        session_id=request.session_id,
        state={
              "user_id": user["user_id"],
              "user_role": user["role"],
              "bu_access": user["bu_access"],
              "downstream_token": credentials.credentials,
          },
    )
    
    content = types.Content(
        role="user",
        parts=[types.Part(text=request.query)],
    )

    response_text = ""
    async for event in runner.run_async(
        user_id=request.user_id,        
        session_id=session.id,
        new_message=content,                
    ):
        if event.is_final_response() and event.content:
            response_text = event.content.parts[0].text
    
    return {"response": response_text, "session_id": request.session_id}
    


if __name__ == "__main__":
    uvicorn.run("agent_api.main:app", host="0.0.0.0", port=8000, reload=False)
