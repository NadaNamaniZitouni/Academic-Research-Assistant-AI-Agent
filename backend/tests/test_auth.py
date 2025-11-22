"""
Unit tests for authentication functionality
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from jose import jwt

from app.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user,
    check_tier_limit
)
from app.models import User


class TestPasswordHashing:
    """Test password hashing and verification"""
    
    def test_hash_password(self):
        """Test password hashing"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert len(hashed) > 0
    
    def test_verify_password_correct(self):
        """Test password verification with correct password"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password"""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = get_password_hash(password)
        
        assert verify_password(wrong_password, hashed) is False


class TestJWTToken:
    """Test JWT token creation and validation"""
    
    @patch('app.auth.SECRET_KEY', 'test-secret-key')
    @patch('app.auth.ALGORITHM', 'HS256')
    def test_create_access_token(self):
        """Test access token creation"""
        data = {"sub": "test-user-id"}
        token = create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    @patch('app.auth.SECRET_KEY', 'test-secret-key')
    @patch('app.auth.ALGORITHM', 'HS256')
    def test_token_contains_user_id(self):
        """Test that token contains user ID"""
        user_id = "test-user-123"
        data = {"sub": user_id}
        token = create_access_token(data)
        
        # Decode token to verify
        decoded = jwt.decode(token, 'test-secret-key', algorithms=['HS256'])
        assert decoded['sub'] == user_id


class TestTierLimits:
    """Test tier-based access control"""
    
    def test_free_tier_limits(self):
        """Test free tier document and query limits"""
        user = Mock(spec=User)
        user.tier = "free"
        user.user_id = "test-user"
        
        db = Mock()
        
        # Mock document count
        doc_mock = Mock()
        doc_mock.count.return_value = 3  # Under limit of 5
        db.query.return_value.filter.return_value = doc_mock
        
        # Should allow (under limit)
        result = check_tier_limit(user, "documents", db)
        assert result is True
        
        # Mock over limit
        doc_mock.count.return_value = 6  # Over limit of 5
        result = check_tier_limit(user, "documents", db)
        assert result is False
    
    def test_pro_tier_unlimited_queries(self):
        """Test pro tier has unlimited queries"""
        user = Mock(spec=User)
        user.tier = "pro"
        user.user_id = "test-user"
        
        db = Mock()
        usage_mock = Mock()
        usage_mock.queries_count = 1000  # High usage
        db.query.return_value.filter.return_value.first.return_value = usage_mock
        
        # Pro tier should allow unlimited queries
        result = check_tier_limit(user, "queries", db)
        assert result is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

