"""
Authentication and authorization utilities
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import os
import bcrypt
from .database import get_db
from .models import User

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-use-env-var")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    try:
        # Use bcrypt directly
        if isinstance(plain_password, str):
            plain_password = plain_password.encode('utf-8')
        if isinstance(hashed_password, str):
            hashed_password = hashed_password.encode('utf-8')
        return bcrypt.checkpw(plain_password, hashed_password)
    except Exception as e:
        print(f"Error verifying password: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt"""
    # Ensure password is bytes
    if isinstance(password, str):
        password = password.encode('utf-8')
    # Truncate if necessary (bcrypt has 72 byte limit)
    if len(password) > 72:
        password = password[:72]
    # Generate salt and hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password, salt)
    # Return as string for storage
    return hashed.decode('utf-8')


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.user_id == user_id).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get the current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    return current_user


def check_tier_limit(user: User, feature: str, db: Session) -> bool:
    """Check if user's tier allows a feature"""
    tier_limits = {
        "free": {
            "documents": 5,
            "queries_per_month": 20,
            "export": False,
            "api_access": False,
        },
        "starter": {
            "documents": 50,
            "queries_per_month": 500,
            "export": True,
            "api_access": False,
        },
        "pro": {
            "documents": 200,
            "queries_per_month": -1,  # unlimited
            "export": True,
            "api_access": True,
        },
        "team": {
            "documents": 1000,
            "queries_per_month": -1,
            "export": True,
            "api_access": True,
        }
    }
    
    limits = tier_limits.get(user.tier, tier_limits["free"])
    
    if feature == "documents":
        # Count user's documents
        from .models import Document
        try:
            doc_count = db.query(Document).filter(Document.user_id == user.user_id).count()
        except Exception as e:
            # If user_id column doesn't exist yet, count all documents (fallback)
            print(f"Warning: Error counting user documents (may need migration): {e}")
            doc_count = db.query(Document).count()
        max_docs = limits["documents"]
        return doc_count < max_docs
    
    elif feature == "queries":
        # Check monthly query limit
        from .models import UserUsage
        current_month = datetime.utcnow().strftime("%Y-%m")
        usage = db.query(UserUsage).filter(
            UserUsage.user_id == user.user_id,
            UserUsage.month == current_month
        ).first()
        
        if usage is None:
            return True  # No usage record, allow
        
        max_queries = limits["queries_per_month"]
        if max_queries == -1:
            return True  # Unlimited
        return usage.queries_count < max_queries
    
    elif feature in ["export", "api_access"]:
        return limits.get(feature, False)
    
    return False

