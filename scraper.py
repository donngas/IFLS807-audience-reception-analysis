import os
import time
import urllib.request
import urllib.parse
import json
import praw
from praw.models import Submission
from typing import List, Optional
from tqdm import tqdm
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

def fetch_json(url: str, use_impersonation: bool = False) -> dict:
    """Helper to fetch JSON from a URL, optionally impersonating a browser using curl_cffi."""
    from curl_cffi import requests
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    
    if use_impersonation:
        headers.update({
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        r = requests.get(url, headers=headers, impersonate="chrome120", timeout=15.0)
    else:
        r = requests.get(url, headers=headers, timeout=15.0)
        
    if r.status_code != 200:
        raise Exception(f"HTTP Error {r.status_code}: {r.text[:200]}")
    return r.json()

def build_pullpush_url(
    endpoint: str,
    encoded_query: str,
    limit: int,
    subreddit: Optional[str] = None,
    sort: str = "top",
    time_filter: str = "all"
) -> str:
    """Build a PullPush API URL with proper sort and time filter translation.
    
    Reddit sort values map to PullPush as follows:
      top        -> sort_type=score&sort=desc
      hot        -> sort_type=score&sort=desc  (best approximation)
      new        -> sort_type=created_utc&sort=desc
      relevance  -> (no sort_type, use PullPush default)
      
    Reddit time_filter values map to PullPush 'after' epoch offsets:
      all   -> no 'after' param
      year  -> after=<now - 365d>
      month -> after=<now - 30d>
      week  -> after=<now - 7d>
      day   -> after=<now - 1d>
    """
    import time as _time
    
    pp_url = f"https://api.pullpush.io/reddit/search/{endpoint}/?q={encoded_query}&size={limit}"
    
    if subreddit:
        pp_url += f"&subreddit={subreddit.strip()}"
    
    # Sort mapping
    sort_map = {
        "top": "sort_type=score&sort=desc",
        "hot": "sort_type=score&sort=desc",
        "new": "sort_type=created_utc&sort=desc",
    }
    if sort in sort_map:
        pp_url += f"&{sort_map[sort]}"
    
    # Time filter mapping
    time_offsets = {
        "day": 86400,
        "week": 604800,
        "month": 2592000,
        "year": 31536000,
    }
    if time_filter in time_offsets:
        after_epoch = int(_time.time()) - time_offsets[time_filter]
        pp_url += f"&after={after_epoch}"
    # "all" -> no 'after' constraint
    
    return pp_url

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
    """Orchestrate scraping using either PRAW, unauthenticated JSON, or Playwright directly."""
    if method == "praw":
        scrape_reddit_praw(query, subreddits, post_limit, comment_limit, sort, time_filter, skip_existing)
    elif method == "json":
        scrape_reddit_json(query, subreddits, post_limit, comment_limit, sort, time_filter, skip_existing)
    elif method == "playwright":
        scrape_reddit_playwright(query, subreddits, post_limit, comment_limit, sort, time_filter, skip_existing)
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

class PlaywrightManager:
    """Manages a lazy-loaded Playwright browser session for persistent browser context."""
    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def get_page(self):
        if self._page is None:
            from playwright.sync_api import sync_playwright
            print("Lazy initializing Playwright browser...")
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
            self._context = self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800}
            )
            self._page = self._context.new_page()
            # Pre-navigate to Reddit homepage to set up cookies for potential fetches
            print("Navigating Playwright to Reddit homepage...")
            try:
                self._page.goto("https://www.reddit.com", wait_until="networkidle", timeout=20000)
            except Exception as e:
                print(f"Warning: Playwright pre-navigation failed: {e}")
        return self._page

    def close(self):
        if self._page:
            try:
                self._page.close()
            except Exception:
                pass
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass


def fetch_json_playwright(page, url: str) -> dict:
    """Fetch JSON from a URL using Playwright (direct navigation or in-context fetch)."""
    import json
    import time
    
    # Approach 1: Direct page.goto
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=15000)
        if response and response.ok:
            try:
                return response.json()
            except Exception:
                pass
        # Browser might render the JSON as text inside a <pre> or body. Let's try evaluating body innerText.
        body_text = page.evaluate("() => document.body.innerText")
        return json.loads(body_text)
    except Exception as e1:
        # Approach 2: Load reddit.com homepage first, then fetch in context
        try:
            if "reddit.com" not in page.url:
                page.goto("https://www.reddit.com", wait_until="networkidle", timeout=20000)
                time.sleep(1)
            json_str = page.evaluate(f"""
                async () => {{
                    const res = await fetch('{url}');
                    if (!res.ok) throw new Error('HTTP status ' + res.status);
                    return await res.text();
                }}
            """)
            return json.loads(json_str)
        except Exception as e2:
            raise Exception(f"Playwright fetch failed: Approach 1: {e1}, Approach 2: {e2}")


def save_json_comments(session, post_id: str, c_payload, comment_limit: int) -> int:
    """Helper to parse comments from a Reddit JSON payload and save them to the DB."""
    if not isinstance(c_payload, list) or len(c_payload) < 2:
        raise ValueError("Invalid Reddit comments JSON payload structure.")
        
    c_children = c_payload[1].get("data", {}).get("children", [])
    comments_fetched = 0
    for c_child in c_children:
        if comments_fetched >= comment_limit:
            break
        if c_child.get("kind") != "t1":
            continue
        c_data = c_child.get("data", {})
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
    return comments_fetched


def fetch_comments_for_post(session, post_id: str, comment_limit: int, pw_manager: PlaywrightManager) -> int:
    """Fetch comments using direct JSON, Playwright, or PullPush fallback."""
    comments_url = f"https://www.reddit.com/comments/{post_id}.json?limit={comment_limit}"
    
    # 1. Direct JSON (curl_cffi)
    try:
        c_payload = fetch_json(comments_url, use_impersonation=True)
        return save_json_comments(session, post_id, c_payload, comment_limit)
    except Exception as e:
        tqdm.write(f"  Direct comments fetch failed for post {post_id}: {e}. Trying Playwright fallback...")
        
    # 2. Playwright fallback
    try:
        page = pw_manager.get_page()
        c_payload = fetch_json_playwright(page, comments_url)
        return save_json_comments(session, post_id, c_payload, comment_limit)
    except Exception as e:
        tqdm.write(f"  Playwright comments fetch failed for post {post_id}: {e}. Falling back to PullPush...")
        
    # 3. PullPush fallback
    return fetch_comments_pullpush(session, post_id, comment_limit)


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
    
    posts_list = []
    
    t_val = time_filter
    if subreddits and len(subreddits) > 0:
        subreddit_str = "+".join([s.strip() for s in subreddits])
        direct_url = f"https://www.reddit.com/r/{subreddit_str}/search.json?q={encoded_query}&sort={sort}&t={t_val}&limit={post_limit}"
    else:
        direct_url = f"https://www.reddit.com/search.json?q={encoded_query}&sort={sort}&t={t_val}&limit={post_limit}"
        
    pw_manager = PlaywrightManager()
    
    # 1. Try direct search using curl_cffi
    print(f"Attempting direct Reddit JSON scrape (with Chrome impersonation)...")
    try:
        payload = fetch_json(direct_url, use_impersonation=True)
        raw_children = payload.get("data", {}).get("children", [])
        posts_list = [child.get("data", {}) for child in raw_children]
        print(f"  SUCCESS! Fetched {len(posts_list)} posts directly from Reddit.")
    except Exception as e:
        print(f"  Direct Reddit search failed: {e}")
        
    # 2. Try Playwright search fallback
    if not posts_list:
        print("Falling back to Playwright for search...")
        try:
            page = pw_manager.get_page()
            payload = fetch_json_playwright(page, direct_url)
            raw_children = payload.get("data", {}).get("children", [])
            posts_list = [child.get("data", {}) for child in raw_children]
            print(f"  SUCCESS! Fetched {len(posts_list)} posts using Playwright.")
        except Exception as e:
            print(f"  Playwright search failed: {e}")
            
    # 3. Try PullPush fallback
    if not posts_list:
        print("Falling back to PullPush API for submissions...")
        subs = subreddits if subreddits else [None]
        for sub in subs:
            pp_url = build_pullpush_url("submission", encoded_query, post_limit, subreddit=sub, sort=sort, time_filter=time_filter)
            print(f"  Querying PullPush: {pp_url}")
            try:
                payload = fetch_json(pp_url, use_impersonation=False)
                returned_data = payload.get("data", [])
                print(f"  PullPush API returned {len(returned_data)} submissions.")
                posts_list.extend(returned_data)
            except Exception as e:
                print(f"  PullPush failed for subreddit {sub}: {e}")
                
    added_posts = 0
    skipped_posts = 0
    try:
        with Session(engine) as session:
            for p_data in tqdm(posts_list, desc="Scraping comments & saving posts"):
                post_id = p_data.get("id")
                if not post_id:
                    continue
                    
                if skip_existing:
                    existing = session.get(Post, post_id)
                    if existing:
                        skipped_posts += 1
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
                added_posts += 1
                
                # Fetch comments using the full direct -> playwright -> pullpush chain
                fetch_comments_for_post(session, post_id, comment_limit, pw_manager)
                
                # Sleep to respect rate limits
                time.sleep(1.0)
    finally:
        pw_manager.close()
        
    print(f"Scraping completed. Added {added_posts} new posts to database, skipped {skipped_posts} existing posts.")


def scrape_reddit_playwright(
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
    
    posts_list = []
    
    t_val = time_filter
    if subreddits and len(subreddits) > 0:
        subreddit_str = "+".join([s.strip() for s in subreddits])
        direct_url = f"https://www.reddit.com/r/{subreddit_str}/search.json?q={encoded_query}&sort={sort}&t={t_val}&limit={post_limit}"
    else:
        direct_url = f"https://www.reddit.com/search.json?q={encoded_query}&sort={sort}&t={t_val}&limit={post_limit}"
        
    pw_manager = PlaywrightManager()
    
    # 1. Playwright search (primary)
    print(f"Attempting Reddit JSON scrape using Playwright...")
    try:
        page = pw_manager.get_page()
        payload = fetch_json_playwright(page, direct_url)
        raw_children = payload.get("data", {}).get("children", [])
        posts_list = [child.get("data", {}) for child in raw_children]
        print(f"  SUCCESS! Fetched {len(posts_list)} posts using Playwright.")
    except Exception as e:
        print(f"  Playwright search failed: {e}")
        
    # 2. Try PullPush fallback
    if not posts_list:
        print("Falling back to PullPush API for submissions...")
        subs = subreddits if subreddits else [None]
        for sub in subs:
            pp_url = build_pullpush_url("submission", encoded_query, post_limit, subreddit=sub, sort=sort, time_filter=time_filter)
            print(f"  Querying PullPush: {pp_url}")
            try:
                payload = fetch_json(pp_url, use_impersonation=False)
                returned_data = payload.get("data", [])
                print(f"  PullPush API returned {len(returned_data)} submissions.")
                posts_list.extend(returned_data)
            except Exception as e:
                print(f"  PullPush failed for subreddit {sub}: {e}")
                
    added_posts = 0
    skipped_posts = 0
    try:
        with Session(engine) as session:
            for p_data in tqdm(posts_list, desc="Scraping comments & saving posts"):
                post_id = p_data.get("id")
                if not post_id:
                    continue
                    
                if skip_existing:
                    existing = session.get(Post, post_id)
                    if existing:
                        skipped_posts += 1
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
                added_posts += 1
                
                # Sleep to respect rate limits
                time.sleep(1.0)
                
                # Fetch comments: Playwright primary, fallback to PullPush
                comments_url = f"https://www.reddit.com/comments/{post_id}.json?limit={comment_limit}"
                comments_fetched = 0
                try:
                    page = pw_manager.get_page()
                    c_payload = fetch_json_playwright(page, comments_url)
                    comments_fetched = save_json_comments(session, post_id, c_payload, comment_limit)
                except Exception as e:
                    tqdm.write(f"  Playwright comments failed for post {post_id}: {e}. Falling back to PullPush.")
                    comments_fetched = fetch_comments_pullpush(session, post_id, comment_limit)
                    
    finally:
        pw_manager.close()
        
    print(f"Scraping completed. Added {added_posts} new posts to database, skipped {skipped_posts} existing posts.")


def fetch_comments_pullpush(session, post_id: str, comment_limit: int) -> int:
    """Helper to fetch comments for a submission from PullPush API."""
    comments_url = f"https://api.pullpush.io/reddit/search/comment/?link_id=t3_{post_id}&size={comment_limit}"
    try:
        c_payload = fetch_json(comments_url, use_impersonation=False)
        comments = c_payload.get("data", [])
        comments_fetched = 0
        for c_data in comments:
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
        return comments_fetched
    except Exception as e:
        tqdm.write(f"  PullPush comments failed for post {post_id} with error: {e}")
    return 0
