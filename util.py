import csv
import json
from database import Session, engine, Post, Comment, TagMapping, get_statistics
from sqlmodel import select

def get_joined_data(session: Session):
    # Retrieve all processed posts
    posts_stmt = select(Post, TagMapping.consolidated_tag).outerjoin(
        TagMapping, Post.raw_tag == TagMapping.raw_tag
    ).where(Post.status == "processed")
    posts_data = session.exec(posts_stmt).all()
    
    # Retrieve all processed comments
    comments_stmt = select(Comment, TagMapping.consolidated_tag).outerjoin(
        TagMapping, Comment.raw_tag == TagMapping.raw_tag
    ).where(Comment.status == "processed")
    comments_data = session.exec(comments_stmt).all()
    
    combined = []
    for post, consolidated in posts_data:
        combined.append({
            "type": "post",
            "id": post.id,
            "parent_id": None,
            "subreddit": post.subreddit,
            "text": f"{post.title} {post.selftext}".strip(),
            "score": post.score,
            "created_utc": post.created_utc,
            "sentiment": post.sentiment,
            "summary": post.summary,
            "raw_tag": post.raw_tag,
            "consolidated_tag": consolidated
        })
        
    for comment, consolidated in comments_data:
        combined.append({
            "type": "comment",
            "id": comment.id,
            "parent_id": comment.post_id,
            "subreddit": None, # Could join post to get subreddit if needed
            "text": comment.body,
            "score": comment.score,
            "created_utc": comment.created_utc,
            "sentiment": comment.sentiment,
            "summary": comment.summary,
            "raw_tag": comment.raw_tag,
            "consolidated_tag": consolidated
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
