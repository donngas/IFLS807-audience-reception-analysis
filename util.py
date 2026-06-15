import csv
import json
from database import Session, engine, Post, Comment, get_statistics, Annotation
from sqlmodel import select

def get_joined_data(session: Session):
    # Retrieve all processed posts along with their annotations
    posts_stmt = select(Post, Annotation).join(
        Annotation, Post.id == Annotation.item_id
    ).where(Post.status == "processed").where(Annotation.item_type == "post")
    posts_data = session.exec(posts_stmt).all()
    
    # Retrieve all processed comments along with their annotations
    comments_stmt = select(Comment, Annotation).join(
        Annotation, Comment.id == Annotation.item_id
    ).where(Comment.status == "processed").where(Annotation.item_type == "comment")
    comments_data = session.exec(comments_stmt).all()
    
    combined = []
    for post, annotation in posts_data:
        combined.append({
            "type": "post",
            "id": post.id,
            "parent_id": None,
            "subreddit": post.subreddit,
            "text": f"{post.title} {post.selftext}".strip(),
            "score": post.score,
            "created_utc": post.created_utc,
            "sentiment": annotation.sentiment,
            "summary": annotation.summary,
            "raw_tag": annotation.raw_tag,
            "consolidated_tag": annotation.consolidated_tag
        })
        
    for comment, annotation in comments_data:
        combined.append({
            "type": "comment",
            "id": comment.id,
            "parent_id": comment.post_id,
            "subreddit": None, # Could join post to get subreddit if needed
            "text": comment.body,
            "score": comment.score,
            "created_utc": comment.created_utc,
            "sentiment": annotation.sentiment,
            "summary": annotation.summary,
            "raw_tag": annotation.raw_tag,
            "consolidated_tag": annotation.consolidated_tag
        })
        
    return combined

def export_to_csv(filepath: str):
    with Session(engine) as session:
        data = get_joined_data(session)
        
    if not data:
        print("No processed data found to export.")
        return
        
    headers = data[0].keys()
    
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Exported {len(data)} items to {filepath}")

def export_to_json(filepath: str):
    with Session(engine) as session:
        data = get_joined_data(session)
        
    if not data:
        print("No processed data found to export.")
        return
        
    with open(filepath, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    print(f"Exported {len(data)} items to {filepath}")

def print_stats():
    with Session(engine) as session:
        stats = get_statistics(session)
        
    print("\n--- Pipeline Statistics ---")
    print("\nPost Status Counts:")
    for status, count in stats.get("post_status_counts", {}).items():
        print(f"  {status}: {count}")
        
    print("\nComment Status Counts:")
    for status, count in stats.get("comment_status_counts", {}).items():
        print(f"  {status}: {count}")
        
    print(f"\nAverage Post Sentiment: {stats.get('average_post_sentiment')}")
    print(f"Average Comment Sentiment: {stats.get('average_comment_sentiment')}")
    print("---------------------------\n")
