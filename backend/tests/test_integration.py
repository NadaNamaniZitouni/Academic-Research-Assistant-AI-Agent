"""
Integration tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os

from app.main import app
from app.database import Base, get_db
from app.models import User
from app.auth import get_password_hash


# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def test_db():
    """Create test database and clean up after"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(test_db):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def test_user(test_db):
    """Create a test user"""
    db = TestingSessionLocal()
    user = User(
        user_id="test-user-123",
        email="test@example.com",
        username="testuser",
        hashed_password=get_password_hash("testpass123"),
        tier="free",
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    yield user
    db.delete(user)
    db.commit()
    db.close()


class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    def test_register_user(self, client, test_db):
        """Test user registration"""
        response = client.post("/auth/register", json={
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "password123",
            "full_name": "New User"
        })
        
        assert response.status_code == 201
        assert "user_id" in response.json()
        assert response.json()["email"] == "newuser@example.com"
    
    def test_register_duplicate_email(self, client, test_db):
        """Test registration with duplicate email"""
        # Register first user
        client.post("/auth/register", json={
            "email": "duplicate@example.com",
            "username": "user1",
            "password": "pass123"
        })
        
        # Try to register with same email
        response = client.post("/auth/register", json={
            "email": "duplicate@example.com",
            "username": "user2",
            "password": "pass123"
        })
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()
    
    def test_login_success(self, client, test_user):
        """Test successful login"""
        response = client.post("/auth/login-json", json={
            "username": "testuser",
            "password": "testpass123"
        })
        
        assert response.status_code == 200
        assert "access_token" in response.json()
        assert "user" in response.json()
    
    def test_login_wrong_password(self, client, test_user):
        """Test login with wrong password"""
        response = client.post("/auth/login-json", json={
            "username": "testuser",
            "password": "wrongpassword"
        })
        
        assert response.status_code == 401
    
    def test_get_current_user(self, client, test_user):
        """Test getting current user info"""
        # First login to get token
        login_response = client.post("/auth/login-json", json={
            "username": "testuser",
            "password": "testpass123"
        })
        token = login_response.json()["access_token"]
        
        # Get current user
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        assert response.json()["username"] == "testuser"


class TestQueryEndpoints:
    """Test query endpoints"""
    
    def test_query_requires_auth(self, client):
        """Test that query endpoint requires authentication"""
        response = client.post("/query", json={
            "query": "test question",
            "k": 5
        })
        
        assert response.status_code == 401
    
    def test_query_with_auth(self, client, test_user):
        """Test query with authentication"""
        # Login
        login_response = client.post("/auth/login-json", json={
            "username": "testuser",
            "password": "testpass123"
        })
        token = login_response.json()["access_token"]
        
        # Query (will fail without documents, but should get past auth)
        response = client.post(
            "/query",
            json={"query": "test question", "k": 5},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Should not be 401 (unauthorized)
        assert response.status_code != 401


class TestUploadEndpoints:
    """Test upload endpoints"""
    
    def test_upload_requires_auth(self, client):
        """Test that upload requires authentication"""
        response = client.post("/upload", files={"file": ("test.pdf", b"fake pdf content", "application/pdf")})
        
        assert response.status_code == 401


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

