import csv
import json
from database import Session, engine, Post, Comment, get_statistics, Annotation
from sqlmodel import select

SENTIMENT_BUCKETS = [-1.0, -0.5, 0.0, 0.5, 1.0]
SENTIMENT_LABELS = {
    -1.0: "Strongly Negative",
    -0.5: "Mildly Negative",
    0.0: "Neutral/Mixed",
    0.5: "Mildly Positive",
    1.0: "Strongly Positive",
}

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
            "is_substantive": annotation.is_substantive,
            "reception_reason": annotation.reception_reason,
            "consolidated_tag": annotation.consolidated_tag,
            "cluster_explanation": annotation.cluster_explanation
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
            "is_substantive": annotation.is_substantive,
            "reception_reason": annotation.reception_reason,
            "consolidated_tag": annotation.consolidated_tag,
            "cluster_explanation": annotation.cluster_explanation
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

def get_sentiment_distribution(session: Session) -> dict[str, dict[float, int]]:
    distribution = {
        "all": {bucket: 0 for bucket in SENTIMENT_BUCKETS},
        "post": {bucket: 0 for bucket in SENTIMENT_BUCKETS},
        "comment": {bucket: 0 for bucket in SENTIMENT_BUCKETS},
    }
    rows = session.exec(select(Annotation.item_type, Annotation.sentiment)).all()
    for item_type, sentiment in rows:
        sentiment = float(sentiment)
        if sentiment not in SENTIMENT_BUCKETS:
            sentiment = min(SENTIMENT_BUCKETS, key=lambda x: abs(x - sentiment))
        distribution["all"][sentiment] += 1
        if item_type in distribution:
            distribution[item_type][sentiment] += 1
    return distribution

def print_sentiment_distribution(distribution: dict[str, dict[float, int]]):
    total = sum(distribution["all"].values())
    print("\nSentiment Score Distribution:")
    if total == 0:
        print("  No sentiment annotations available yet.")
        return

    max_count = max(distribution["all"].values()) or 1
    for bucket in SENTIMENT_BUCKETS:
        count = distribution["all"][bucket]
        bar_len = int((count / max_count) * 28) if count else 0
        bar = "#" * bar_len
        percent = (count / total) * 100
        print(f"  {bucket:>4.1f} {SENTIMENT_LABELS[bucket]:<18} | {bar:<28} {count:>4} ({percent:>5.1f}%)")

    print("\nBy Item Type:")
    for item_type in ["post", "comment"]:
        item_total = sum(distribution[item_type].values())
        if item_total:
            values = ", ".join(
                f"{bucket:g}: {distribution[item_type][bucket]} ({distribution[item_type][bucket] / item_total * 100:.1f}%)"
                for bucket in SENTIMENT_BUCKETS
            )
        else:
            values = ", ".join(f"{bucket:g}: 0 (0.0%)" for bucket in SENTIMENT_BUCKETS)
        print(f"  {item_type.title():<7} ({item_total}): {values}")

def get_substance_distribution(session: Session) -> dict[str, int]:
    rows = session.exec(select(Annotation.is_substantive)).all()
    substantive = sum(1 for value in rows if value)
    reaction_only = len(rows) - substantive
    return {
        "substantive": substantive,
        "reaction_only": reaction_only,
        "total": len(rows),
    }

def print_stats():
    with Session(engine) as session:
        stats = get_statistics(session)
        sentiment_distribution = get_sentiment_distribution(session)
        substance_distribution = get_substance_distribution(session)
        
    print("\n--- Pipeline Statistics ---")
    print("\nPost Status Counts:")
    for status, count in stats.get("post_status_counts", {}).items():
        print(f"  {status}: {count}")
        
    print("\nComment Status Counts:")
    for status, count in stats.get("comment_status_counts", {}).items():
        print(f"  {status}: {count}")
        
    print(f"\nAverage Post Sentiment: {stats.get('average_post_sentiment')}")
    print(f"Average Comment Sentiment: {stats.get('average_comment_sentiment')}")
    total_substance = substance_distribution["total"]
    print("\nSubstance Classification:")
    if total_substance:
        print(f"  Substantive:   {substance_distribution['substantive']} ({substance_distribution['substantive'] / total_substance * 100:.1f}%)")
        print(f"  Reaction Only: {substance_distribution['reaction_only']} ({substance_distribution['reaction_only'] / total_substance * 100:.1f}%)")
    else:
        print("  No annotations available yet.")
    print_sentiment_distribution(sentiment_distribution)
    print("---------------------------\n")
