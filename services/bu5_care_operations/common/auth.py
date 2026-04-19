from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt as pyjwt
from common.config import settings

security=HTTPBearer(auto_error=False)

async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing token")
    
    try:
        return pyjwt.decode(credentials.credentials, settings.jwt_secret, algorithms=["HS256"])
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
def require_bu(*allowed_bus: str):
    async def checker(user: dict = Depends(get_current_user)):
        if user["role"] == "admin":
            return user
        user_bus = set(user.get("bu_access", []))
        required_bus = set(allowed_bus)                                            
        if not required_bus.intersection(user_bus):
            raise HTTPException(status_code=403, detail=f"No access to {allowed_bus}")
        return user
    return checker

def require_role(*allowed_roles: str):
    async def checker(user: dict = Depends(get_current_user)):
        if user["role"] == "admin":
            return user
        user_roles = set(user.get("role", []))
        require_roles = set(allowed_roles)
        if not require_roles.intersection(user_roles):
            raise HTTPException(status_code=401, detail="No access to operation")
        return user
    return checker