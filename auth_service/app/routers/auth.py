from fastapi import APIRouter, HTTPException
from models.user import UserCreate, UserLogin
from services.auth_service import create_user, authenticate_user

router = APIRouter()

@router.post("/register")
async def register(user: UserCreate):
    try:
        create_user(user)
        return {"message": "User registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login")
async def login(user: UserLogin):
    user_data = authenticate_user(user.email, user.password)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "user": user_data}
