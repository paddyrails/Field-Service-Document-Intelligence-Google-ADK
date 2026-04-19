from motor.motor_asyncio import AsyncIOMotorClient

from common.config import settings

_client = AsyncIOMotorClient(settings.mongodb_uri)
_db = _client[settings.mongodb_db_name]
_users = _db["users"]


async def find_by_email(email: str) -> dict | None:
    doc = await _users.find_one({"email": email})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc


async def create_user(user: dict) -> dict:
    result = await _users.insert_one(user)
    user["_id"] = str(result.inserted_id)
    return user


async def ensure_indexes():
    await _users.create_index("email", unique=True)
