from fastapi import HTTPException

from common.security import hash_password, verify_password, create_access_token
from dao import user_dao


async def register(email: str, password: str, full_name: str, role: str, bu_access: list[str]) -> dict:
    existing = await user_dao.find_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = await user_dao.create_user({
        "email": email,
        "password_hash": hash_password(password),
        "full_name": full_name,
        "role": role,
        "bu_access": bu_access,
    })

    return {"user_id": user["_id"], "email": email, "role": role}


async def login(email: str, password: str) -> dict:
    user = await user_dao.find_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({
        "sub": user["_id"],
        "email": user["email"],
        "role": user["role"],
        "bu_access": user["bu_access"],
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "full_name": user["full_name"],
    }
