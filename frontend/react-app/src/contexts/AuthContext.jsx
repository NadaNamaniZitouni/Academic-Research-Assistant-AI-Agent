import { createContext, useContext, useState, useEffect } from 'react';
import { login, register, getCurrentUser } from '../services/api';

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in on mount
    if (token) {
      fetchUser();
    } else {
      setLoading(false);
    }
  }, []);

  const fetchUser = async () => {
    try {
      const userData = await getCurrentUser();
      setUser(userData);
    } catch (error) {
      // Token might be invalid, clear it
      localStorage.removeItem('token');
      setToken(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async (username, password) => {
    try {
      const response = await login(username, password);
      const { access_token, user: userData } = response;
      
      localStorage.setItem('token', access_token);
      setToken(access_token);
      setUser(userData);
      
      return { success: true };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Login failed. Please check your credentials.'
      };
    }
  };

  const handleRegister = async (email, username, password, fullName) => {
    try {
      await register(email, username, password, fullName);
      // Auto-login after registration
      return await handleLogin(username, password);
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Registration failed. Please try again.'
      };
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
  };

  const value = {
    user,
    token,
    loading,
    login: handleLogin,
    register: handleRegister,
    logout: handleLogout,
    isAuthenticated: !!token && !!user
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

