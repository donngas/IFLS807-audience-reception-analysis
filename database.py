import os
from typing import Optional, List
from sqlmodel import Field, SQLModel, Session, create_engine, select, func

# Database Schema

class Post(SQLModel, table=True):
    id: str = Field(primary_key=True, description="Reddit Post ID")
    subreddit: str
    title: str
    selftext: str
    score: int
    created_utc: float
    status: str = Field(default="pending", description="pending | processed | failed | skipped")
    sentiment: Optional[float] = None
    summary: Optional[str] = None
    raw_tag: Optional[str] = None

class Comment(SQLModel, table=True):
    id: str = Field(primary_key=True, description="Reddit Comment ID")
    post_id: str = Field(foreign_key="post.id")
    body: str
    score: int
    created_utc: float
    status: str = Field(default="pending", description="pending | processed | failed | skipped")
    sentiment: Optional[float] = None
    summary: Optional[str] = None
    raw_tag: Optional[str] = None

class TagMapping(SQLModel, table=True):
    raw_tag: str = Field(primary_key=True, description="Primary Key")
    consolidated_tag: Optional[str] = None


# Database Connection & Setup

sqlite_file_name = "audience_reception.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=False)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# CRUD Operations

def upsert_post(session: Session, post: Post):
    """Upsert a post using session.merge()"""
    session.merge(post)
    session.commit()

def upsert_comment(session: Session, comment: Comment):
    """Upsert a comment using session.merge()"""
    session.merge(comment)
    session.commit()

def get_pending_posts(session: Session, limit: int = 50) -> List[Post]:
    statement = select(Post).where(Post.status == "pending").limit(limit)
    return session.exec(statement).all()

def get_pending_comments(session: Session, limit: int = 50) -> List[Comment]:
    statement = select(Comment).where(Comment.status == "pending").limit(limit)
    return session.exec(statement).all()

def get_unmapped_raw_tags(session: Session) -> List[str]:
    statement = select(TagMapping.raw_tag).where(TagMapping.consolidated_tag == None)
    return session.exec(statement).all()

def upsert_tag_mapping(session: Session, tag: str, consolidated: Optional[str] = None):
    # Upsert logic to not overwrite consolidated if we are just adding the raw_tag
    existing = session.get(TagMapping, tag)
    if not existing:
        new_tag = TagMapping(raw_tag=tag, consolidated_tag=consolidated)
        session.add(new_tag)
    else:
        if consolidated is not None:
            existing.consolidated_tag = consolidated
            session.add(existing)
    session.commit()

def get_statistics(session: Session) -> dict:
    post_stats = session.exec(select(Post.status, func.count(Post.id)).group_by(Post.status)).all()
    comment_stats = session.exec(select(Comment.status, func.count(Comment.id)).group_by(Comment.status)).all()
    
    avg_post_sentiment = session.exec(select(func.avg(Post.sentiment)).where(Post.sentiment != None)).first()
    avg_comment_sentiment = session.exec(select(func.avg(Comment.sentiment)).where(Comment.sentiment != None)).first()
    
    return {
        "post_status_counts": dict(post_stats),
        "comment_status_counts": dict(comment_stats),
        "average_post_sentiment": avg_post_sentiment,
        "average_comment_sentiment": avg_comment_sentiment
    }
