"""
Analytics and statistics endpoints
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from typing import List, Dict
from .models import QueryHistory, Document, User, UserUsage


def get_query_analytics(
    db: Session,
    user_id: str,
    days: int = 30
) -> Dict:
    """Get query analytics for a user"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Total queries
    total_queries = db.query(QueryHistory).filter(
        QueryHistory.user_id == user_id,
        QueryHistory.created_at >= start_date
    ).count()
    
    # Average response time
    avg_response_time = db.query(func.avg(QueryHistory.response_time_ms)).filter(
        QueryHistory.user_id == user_id,
        QueryHistory.created_at >= start_date
    ).scalar() or 0
    
    # Most asked questions
    most_asked = db.query(
        QueryHistory.query_text,
        func.count(QueryHistory.id).label('count')
    ).filter(
        QueryHistory.user_id == user_id,
        QueryHistory.created_at >= start_date
    ).group_by(QueryHistory.query_text).order_by(desc('count')).limit(10).all()
    
    # Queries per day (last 30 days)
    queries_per_day = db.query(
        func.date(QueryHistory.created_at).label('date'),
        func.count(QueryHistory.id).label('count')
    ).filter(
        QueryHistory.user_id == user_id,
        QueryHistory.created_at >= start_date
    ).group_by(func.date(QueryHistory.created_at)).order_by('date').all()
    
    # Response time distribution
    response_times = db.query(QueryHistory.response_time_ms).filter(
        QueryHistory.user_id == user_id,
        QueryHistory.created_at >= start_date,
        QueryHistory.response_time_ms.isnot(None)
    ).all()
    
    response_time_list = [rt[0] for rt in response_times if rt[0] is not None]
    
    # Documents queried
    documents_queried = db.query(
        QueryHistory.doc_id,
        func.count(QueryHistory.id).label('count')
    ).filter(
        QueryHistory.user_id == user_id,
        QueryHistory.created_at >= start_date,
        QueryHistory.doc_id.isnot(None)
    ).group_by(QueryHistory.doc_id).order_by(desc('count')).limit(10).all()
    
    return {
        "total_queries": total_queries,
        "average_response_time_ms": round(float(avg_response_time), 2) if avg_response_time else 0,
        "most_asked_questions": [
            {"query": q[0], "count": q[1]} for q in most_asked
        ],
        "queries_per_day": [
            {"date": str(q[0]), "count": q[1]} for q in queries_per_day
        ],
        "response_time_stats": {
            "min": min(response_time_list) if response_time_list else 0,
            "max": max(response_time_list) if response_time_list else 0,
            "median": sorted(response_time_list)[len(response_time_list)//2] if response_time_list else 0
        },
        "documents_queried": [
            {"doc_id": d[0], "count": d[1]} for d in documents_queried
        ],
        "period_days": days
    }


def get_user_statistics(
    db: Session,
    user_id: str
) -> Dict:
    """Get overall user statistics"""
    # Document count
    doc_count = db.query(Document).filter(Document.user_id == user_id).count()
    
    # Total chunks - count chunks from user's documents
    from .models import ChunkMetadata
    try:
        # Get all document IDs for this user
        user_doc_ids = db.query(Document.doc_id).filter(
            Document.user_id == user_id
        ).all()
        doc_ids = [doc[0] for doc in user_doc_ids]
        
        if doc_ids:
            chunk_count = db.query(ChunkMetadata).filter(
                ChunkMetadata.doc_id.in_(doc_ids)
            ).count()
        else:
            chunk_count = 0
    except Exception as e:
        # Fallback: if join fails, count all chunks (shouldn't happen but safe)
        print(f"Warning: Error counting chunks for user {user_id}: {e}")
        chunk_count = 0
    
    # Total queries (all time)
    total_queries = db.query(QueryHistory).filter(QueryHistory.user_id == user_id).count()
    
    # Current month usage
    current_month = datetime.utcnow().strftime("%Y-%m")
    usage = db.query(UserUsage).filter(
        UserUsage.user_id == user_id,
        UserUsage.month == current_month
    ).first()
    
    queries_this_month = usage.queries_count if usage else 0
    
    # Get user tier
    user = db.query(User).filter(User.user_id == user_id).first()
    tier = user.tier if user else "free"
    
    return {
        "documents_count": doc_count,
        "total_chunks": chunk_count,
        "total_queries": total_queries,
        "queries_this_month": queries_this_month,
        "tier": tier,
        "account_created": user.created_at.isoformat() if user else None
    }

