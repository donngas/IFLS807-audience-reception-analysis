import os
import time
import urllib.request
import urllib.parse
import json
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

def fetch_unauthenticated_json(url: str) -> dict:
    """Helper to fetch unauthenticated JSON from Reddit with a proper User-Agent."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def scrape_reddit(
    query: str, 
    subreddits: Optional[List[str]] = None, 
    post_limit: int = 100, 
    comment_limit: int = 100, 
    sort: str = "top", 
    time_filter: str = "all", 
    skip_existing: bool = True,
    method: str = "praw"
):
    """Orchestrate scraping using either PRAW or unauthenticated JSON."""
    if method == "praw":
        scrape_reddit_praw(query, subreddits, post_limit, comment_limit, sort, time_filter, skip_existing)
    elif method == "json":
        scrape_reddit_json(query, subreddits, post_limit, comment_limit, sort, time_filter, skip_existing)
    else:
        raise ValueError(f"Unknown data acquisition method: {method}")

def scrape_reddit_praw(
    query: str, 
    subreddits: Optional[List[str]] = None, 
    post_limit: int = 100, 
    comment_limit: int = 100, 
    sort: str = "top", 
    time_filter: str = "all", 
    skip_existing: bool = True
):
    reddit = get_praw_reddit()
    translated_query = translate_query(query)
    
    if subreddits and len(subreddits) > 0:
        subreddit_str = "+".join([s.strip() for s in subreddits])
        target_subreddit = reddit.subreddit(subreddit_str)
    else:
        target_subreddit = reddit.subreddit("all")
        
    print(f"Scraping via PRAW: '{translated_query}' in r/{target_subreddit.display_name} (Sort: {sort}, Time: {time_filter})...")
    
    with Session(engine) as session:
        for submission in target_subreddit.search(translated_query, sort=sort, time_filter=time_filter, limit=post_limit):
            if skip_existing:
                existing = session.get(Post, submission.id)
                if existing:
                    print(f"Skipping already scraped post: {submission.id}")
                    continue
            
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
            
            submission.comments.replace_more(limit=0)
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

def scrape_reddit_json(
    query: str, 
    subreddits: Optional[List[str]] = None, 
    post_limit: int = 100, 
    comment_limit: int = 100, 
    sort: str = "top", 
    time_filter: str = "all", 
    skip_existing: bool = True
):
    translated_query = translate_query(query)
    encoded_query = urllib.parse.quote(translated_query)
    
    t_val = time_filter
    
    if subreddits and len(subreddits) > 0:
        subreddit_str = "+".join([s.strip() for s in subreddits])
        url = f"https://www.reddit.com/r/{subreddit_str}/search.json?q={encoded_query}&sort={sort}&t={t_val}&limit={post_limit}"
    else:
        url = f"https://www.reddit.com/search.json?q={encoded_query}&sort={sort}&t={t_val}&limit={post_limit}"
        
    print(f"Scraping via Unauthenticated JSON: '{translated_query}' (Sort: {sort}, Time: {time_filter})...")
    
    try:
        search_data = fetch_unauthenticated_json(url)
    except Exception as e:
        print(f"Failed to fetch search results via unauthenticated JSON: {e}")
        return
        
    posts_list = search_data.get("data", {}).get("children", [])
    
    with Session(engine) as session:
        for p_item in posts_list:
            p_data = p_item.get("data", {})
            post_id = p_data.get("id")
            if not post_id:
                continue
                
            if skip_existing:
                existing = session.get(Post, post_id)
                if existing:
                    print(f"Skipping already scraped post: {post_id}")
                    continue
                    
            post_author = p_data.get("author", "")
            selftext = p_data.get("selftext", "")
            post_status = "skipped" if is_low_quality(post_author, selftext) else "pending"
            
            post = Post(
                id=post_id,
                subreddit=p_data.get("subreddit", ""),
                title=p_data.get("title", ""),
                selftext=selftext,
                score=p_data.get("score", 0),
                created_utc=p_data.get("created_utc", 0.0),
                status=post_status
            )
            upsert_post(session, post)
            
            # Gentle sleep to respect unauthenticated rate limit
            time.sleep(1.0)
            
            # Fetch comments
            comments_url = f"https://www.reddit.com/comments/{post_id}.json?limit={comment_limit}"
            try:
                comments_payload = fetch_unauthenticated_json(comments_url)
                if len(comments_payload) > 1:
                    comments_list = comments_payload[1].get("data", {}).get("children", [])
                    comments_fetched = 0
                    for c_item in comments_list:
                        if comments_fetched >= comment_limit:
                            break
                        if c_item.get("kind") != "t1":
                            continue
                            
                        c_data = c_item.get("data", {})
                        c_id = c_data.get("id")
                        c_body = c_data.get("body", "")
                        c_author = c_data.get("author", "")
                        
                        comment_status = "skipped" if is_low_quality(c_author, c_body) else "pending"
                        
                        db_comment = Comment(
                            id=c_id,
                            post_id=post_id,
                            body=c_body,
                            score=c_data.get("score", 0),
                            created_utc=c_data.get("created_utc", 0.0),
                            status=comment_status
                        )
                        upsert_comment(session, db_comment)
                        comments_fetched += 1
            except Exception as e:
                print(f"Failed to fetch comments for post {post_id}: {e}")
                
    print("Scraping completed.")
