import os
import praw
from praw.models import Submission
from typing import List, Optional
from database import Session, engine, Post, Comment, upsert_post, upsert_comment

def get_praw_reddit():
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT")
    
    if not all([client_id, client_secret, user_agent]):
        raise ValueError("Missing Reddit API credentials in environment variables (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT).")
        
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

def translate_query(query: str) -> str:
    """Replace { and } with ( and ) for PRAW search."""
    return query.replace("{", "(").replace("}", ")")

def is_low_quality(author_name: str, body_text: str) -> bool:
    """Check if content is from a bot or is deleted/removed."""
    author_name = str(author_name).lower() if author_name else ""
    body_text = str(body_text).lower() if body_text else ""
    
    if author_name == "automoderator" or "bot" in author_name:
        return True
    if body_text in ["[deleted]", "[removed]", ""]:
        return True
    return False

def scrape_reddit(query: str, subreddits: Optional[List[str]] = None, post_limit: int = 100, comment_limit: int = 100, sort: str = "top", time_filter: str = "all", skip_existing: bool = True):
    reddit = get_praw_reddit()
    translated_query = translate_query(query)
    
    if subreddits and len(subreddits) > 0:
        # Join multiple subreddits with +
        subreddit_str = "+".join([s.strip() for s in subreddits])
        target_subreddit = reddit.subreddit(subreddit_str)
    else:
        target_subreddit = reddit.subreddit("all")
        
    print(f"Scraping '{translated_query}' in r/{target_subreddit.display_name} (Sort: {sort}, Time: {time_filter})...")
    
    with Session(engine) as session:
        for submission in target_subreddit.search(translated_query, sort=sort, time_filter=time_filter, limit=post_limit):
            # Check if post exists and skip_existing is True
            if skip_existing:
                existing = session.get(Post, submission.id)
                if existing:
                    print(f"Skipping already scraped post: {submission.id}")
                    continue
            
            # Process Post
            post_author = submission.author.name if submission.author else ""
            post_status = "skipped" if is_low_quality(post_author, submission.selftext) else "pending"
            
            post = Post(
                id=submission.id,
                subreddit=submission.subreddit.display_name,
                title=submission.title,
                selftext=submission.selftext,
                score=submission.score,
                created_utc=submission.created_utc,
                status=post_status
            )
            upsert_post(session, post)
            
            submission.comments.replace_more(limit=0) # Only top-level, limit=0 removes MoreComments
            comments_fetched = 0
            for comment in submission.comments:
                if comments_fetched >= comment_limit:
                    break
                
                comment_author = comment.author.name if comment.author else ""
                comment_status = "skipped" if is_low_quality(comment_author, comment.body) else "pending"
                
                db_comment = Comment(
                    id=comment.id,
                    post_id=submission.id,
                    body=comment.body,
                    score=comment.score,
                    created_utc=comment.created_utc,
                    status=comment_status
                )
                upsert_comment(session, db_comment)
                comments_fetched += 1
                
    print("Scraping completed.")
