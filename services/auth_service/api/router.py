from fastapi import APIRouter

from api.schemas import RegisterRequest, LoginRequest, TokenResponse
from service import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=201)
async def register(body: RegisterRequest) -> dict:
    return await auth_service.register(
        email=body.email,
        password=body.password,
        full_name=body.full_name,
        role=body.role,
        bu_access=body.bu_access,
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    return await auth_service.login(email=body.email, password=body.password)
