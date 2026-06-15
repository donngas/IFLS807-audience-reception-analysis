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

class Comment(SQLModel, table=True):
    id: str = Field(primary_key=True, description="Reddit Comment ID")
    post_id: str = Field(foreign_key="post.id")
    body: str
    score: int
    created_utc: float
    status: str = Field(default="pending", description="pending | processed | failed | skipped")

class Annotation(SQLModel, table=True):
    item_id: str = Field(primary_key=True, description="Reddit Post ID or Comment ID")
    item_type: str = Field(description="post | comment")
    sentiment: float
    summary: str
    raw_tag: str
    consolidated_tag: Optional[str] = Field(default=None, nullable=True)
    cluster_id: Optional[int] = Field(default=None, nullable=True)
    embedding: Optional[str] = Field(default=None, description="JSON serialized array of floats")


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

def upsert_annotation(session: Session, annotation: Annotation):
    """Upsert an annotation using session.merge()"""
    session.merge(annotation)
    session.commit()

def get_pending_posts(session: Session, limit: int = 50) -> List[Post]:
    statement = select(Post).where(Post.status == "pending").limit(limit)
    return session.exec(statement).all()

def get_pending_comments(session: Session, limit: int = 50) -> List[Comment]:
    statement = select(Comment).where(Comment.status == "pending").limit(limit)
    return session.exec(statement).all()

def get_unmapped_raw_tags(session: Session) -> List[str]:
    statement = select(Annotation.raw_tag).where(Annotation.consolidated_tag == None).distinct()
    return session.exec(statement).all()

def update_consolidated_tags(session: Session, tag_mappings: dict[str, str]):
    """Update consolidated_tag in Annotation table for all matching raw_tags"""
    for raw, consolidated in tag_mappings.items():
        statement = select(Annotation).where(Annotation.raw_tag == raw)
        annotations = session.exec(statement).all()
        for ann in annotations:
            ann.consolidated_tag = consolidated
            session.add(ann)
    session.commit()

def reset_failed_items(session: Session, item_type: str):
    """Reset status of failed posts or comments back to pending"""
    if item_type == "post":
        statement = select(Post).where(Post.status == "failed")
        items = session.exec(statement).all()
        for item in items:
            item.status = "pending"
            session.add(item)
    elif item_type == "comment":
        statement = select(Comment).where(Comment.status == "failed")
        items = session.exec(statement).all()
        for item in items:
            item.status = "pending"
            session.add(item)
    session.commit()

def reset_processed_items(session: Session, item_type: str):
    """Reset status of processed posts or comments back to pending and remove their annotations"""
    if item_type == "post":
        posts_stmt = select(Post).where(Post.status == "processed")
        posts = session.exec(posts_stmt).all()
        for p in posts:
            p.status = "pending"
            session.add(p)
            ann = session.get(Annotation, p.id)
            if ann:
                session.delete(ann)
    elif item_type == "comment":
        comments_stmt = select(Comment).where(Comment.status == "processed")
        comments = session.exec(comments_stmt).all()
        for c in comments:
            c.status = "pending"
            session.add(c)
            ann = session.get(Annotation, c.id)
            if ann:
                session.delete(ann)
    session.commit()

def clear_stage_2_clustering(session: Session):
    """Wipe cluster assignments and consolidated tags from all annotations"""
    statement = select(Annotation)
    annotations = session.exec(statement).all()
    for ann in annotations:
        ann.cluster_id = None
        ann.consolidated_tag = None
        session.add(ann)
    session.commit()

def reset_specific_items(session: Session, item_ids: List[str]):
    """Reset specific posts or comments by ID back to pending and remove annotations if present"""
    for item_id in item_ids:
        post = session.get(Post, item_id)
        if post:
            post.status = "pending"
            session.add(post)
            ann = session.get(Annotation, item_id)
            if ann:
                session.delete(ann)
        comment = session.get(Comment, item_id)
        if comment:
            comment.status = "pending"
            session.add(comment)
            ann = session.get(Annotation, item_id)
            if ann:
                session.delete(ann)
    session.commit()

def nuke_database(session: Session):
    """Delete all rows from all tables"""
    from sqlmodel import delete
    session.exec(delete(Annotation))
    session.exec(delete(Comment))
    session.exec(delete(Post))
    session.commit()

def get_statistics(session: Session) -> dict:
    post_stats = session.exec(select(Post.status, func.count(Post.id)).group_by(Post.status)).all()
    comment_stats = session.exec(select(Comment.status, func.count(Comment.id)).group_by(Comment.status)).all()
    
    avg_post_sentiment = session.exec(
        select(func.avg(Annotation.sentiment))
        .where(Annotation.item_type == "post")
    ).first()
    avg_comment_sentiment = session.exec(
        select(func.avg(Annotation.sentiment))
        .where(Annotation.item_type == "comment")
    ).first()
    
    return {
        "post_status_counts": dict(post_stats),
        "comment_status_counts": dict(comment_stats),
        "average_post_sentiment": avg_post_sentiment,
        "average_comment_sentiment": avg_comment_sentiment
    }
