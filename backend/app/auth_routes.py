"""
Authentication routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid

from .database import get_db
from .models import User
from .auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_active_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from .schemas import UserCreate, UserLogin, UserResponse, Token, UsageStats

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.email == user_data.email) | (User.username == user_data.username)
    ).first()
    
    if existing_user:
        if existing_user.email == user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Create new user
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)
    
    new_user = User(
        user_id=user_id,
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        tier="free",
        is_active=True,
        is_premium=False,
        created_at=datetime.utcnow()
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return UserResponse(
        user_id=new_user.user_id,
        email=new_user.email,
        username=new_user.username,
        full_name=new_user.full_name,
        tier=new_user.tier,
        is_premium=new_user.is_premium,
        created_at=new_user.created_at
    )


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login and get access token"""
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == form_data.username) | (User.email == form_data.username)
    ).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_id},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            user_id=user.user_id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            tier=user.tier,
            is_premium=user.is_premium,
            created_at=user.created_at
        )
    )


@router.post("/login-json", response_model=Token)
async def login_json(
    login_data: UserLogin,
    db: Session = Depends(get_db)
):
    """Login using JSON (alternative to OAuth2 form)"""
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == login_data.username) | (User.email == login_data.username)
    ).first()
    
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_id},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            user_id=user.user_id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            tier=user.tier,
            is_premium=user.is_premium,
            created_at=user.created_at
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information"""
    return UserResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        tier=current_user.tier,
        is_premium=current_user.is_premium,
        created_at=current_user.created_at
    )


@router.get("/usage", response_model=UsageStats)
async def get_usage_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current user's usage statistics"""
    from .models import Document, UserUsage
    
    # Count documents
    doc_count = db.query(Document).filter(Document.user_id == current_user.user_id).count()
    
    # Get monthly query usage
    current_month = datetime.utcnow().strftime("%Y-%m")
    usage = db.query(UserUsage).filter(
        UserUsage.user_id == current_user.user_id,
        UserUsage.month == current_month
    ).first()
    
    queries_this_month = usage.queries_count if usage else 0
    
    # Get tier limits
    tier_limits = {
        "free": {"documents": 5, "queries": 20, "export": False, "api": False},
        "starter": {"documents": 50, "queries": 500, "export": True, "api": False},
        "pro": {"documents": 200, "queries": -1, "export": True, "api": True},
        "team": {"documents": 1000, "queries": -1, "export": True, "api": True}
    }
    
    limits = tier_limits.get(current_user.tier, tier_limits["free"])
    
    return UsageStats(
        user_id=current_user.user_id,
        tier=current_user.tier,
        documents_count=doc_count,
        queries_this_month=queries_this_month,
        queries_limit=limits["queries"],
        documents_limit=limits["documents"],
        can_export=limits["export"],
        can_use_api=limits["api"]
    )

