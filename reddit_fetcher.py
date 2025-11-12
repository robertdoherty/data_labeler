# data_fetchers/reddit_fetcher.py
# Contains functions for initializing PRAW and fetching data from Reddit.

import praw
import prawcore # Import specific exceptions
from datetime import datetime, timezone
import logging
from typing import List, Dict, Optional, Any
import json
import html


# <<< Import tenacity for retries
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
# >>> END NEW ADDITION

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s')

#  Define a retry decorator for PRAW API calls
# This decorator will automatically retry functions if they fail due to network issues or Reddit server errors.
retry_on_reddit_api_error = retry(
    wait=wait_random_exponential(multiplier=2, max=60),  # Wait 2s, then 4s, up to 60s
    stop=stop_after_attempt(5),
    retry=(
        retry_if_exception_type(prawcore.exceptions.RequestException) | # For timeouts, connection errors
        retry_if_exception_type(prawcore.exceptions.ServerError)       # For 5xx errors from Reddit
    )
)


# <<< Create decorated helper functions for API calls.
# This is the simplest way to add retries to the specific PRAW actions.
@retry_on_reddit_api_error
def get_new_submissions_with_retry(subreddit_instance, limit):
    """Fetches new submissions with retry logic."""
    logging.debug(f"Fetching .new(limit={limit}) from r/{subreddit_instance.display_name}")
    return list(subreddit_instance.new(limit=limit))


@retry_on_reddit_api_error
def get_comments_with_retry(submission):
    """Replaces 'more comments' and fetches the full comment list with retry logic."""
    submission.comments.replace_more(limit=0) # Fetch all comments
    return submission.comments.list()

@retry_on_reddit_api_error
def search_submissions_with_retry(subreddit_instance, query: str, sort: str = "new", syntax: str = "cloudsearch", limit: Optional[int] = None):
    """
    Perform a subreddit search with retry logic. Useful for time-window pagination using CloudSearch syntax.
    """
    logging.debug(f"Searching r/{subreddit_instance.display_name} with query='{query}', sort='{sort}', syntax='{syntax}', limit={limit}")
    return list(subreddit_instance.search(query=query, sort=sort, syntax=syntax, limit=limit))
# >>> END NEW ADDITION



# -------------------------------------------------
# Image URL helpers (simple and robust for PRAW)
# -------------------------------------------------
def _is_image_url(url: Optional[str]) -> bool:
    if not url:
        return False
    base = url.lower().split("?")[0]
    return base.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))


def _html_unescape(u: Optional[str]) -> Optional[str]:
    return html.unescape(u) if isinstance(u, str) else u


def _dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_image_urls_from_submission(submission) -> List[str]:
    """
    Extract canonical image URLs from a submission using PRAW fields.
    Covers direct image links, preview images, and galleries.
    """
    urls: List[str] = []

    # 1) Direct image link
    try:
        direct_url = getattr(submission, "url", None)
        if _is_image_url(direct_url):
            urls.append(direct_url)
    except Exception:
        pass

    # 2) Preview full-size image
    try:
        prev = getattr(submission, "preview", None)
        if prev and isinstance(prev, dict):
            images = prev.get("images") or []
            if images:
                src = images[0].get("source") or {}
                u = src.get("url")
                if u:
                    urls.append(_html_unescape(u))
    except Exception:
        pass

    # 3) Gallery items
    try:
        if getattr(submission, "is_gallery", False):
            md = getattr(submission, "media_metadata", {}) or {}
            gd = getattr(submission, "gallery_data", {}) or {}
            for it in gd.get("items", []):
                mid = it.get("media_id")
                if not mid:
                    continue
                m = md.get(mid) or {}
                s = (m.get("s") or {})
                u = s.get("u") or s.get("gif") or s.get("mp4")
                if u and _is_image_url(u):
                    urls.append(_html_unescape(u))
    except Exception:
        pass

    return _dedupe_preserve_order(urls)


def initialize_reddit(client_id: str, client_secret: str, user_agent: str, request_timeout: int = 30) -> Optional[praw.Reddit]:
    """Initialize Reddit client with proper error handling."""
    if not client_id or "YOUR_REDDIT_CLIENT_ID_HERE" in client_id:
        logging.error("Reddit Client ID is missing or is a placeholder.")
        return None
    if not client_secret or "YOUR_REDDIT_CLIENT_SECRET_HERE" in client_secret:
        logging.error("Reddit Client Secret is missing or is a placeholder.")
        return None
    if not user_agent:
        logging.error("Reddit User Agent is missing.")
        return None
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            read_only=True,
            request_timeout=request_timeout
        )
        reddit.user.me()
        logging.info(f"Successfully initialized PRAW Reddit instance with user agent: {user_agent}")
        return reddit
    except Exception as e:
        logging.error(f"An error occurred during PRAW initialization: {e}", exc_info=True)
    return None

def fetch_subreddit_posts_with_comments(
    reddit: praw.Reddit,
    subreddit_names: List[str],
    limit_per_subreddit: int = 100,
    include_stickied: bool = False
) -> Dict[str, Any]:
    """
    Fetch posts and comments from specified subreddits and return in nested JSON format.
    
    Args:
        reddit: Initialized PRAW Reddit client
        subreddit_names: List of subreddit names to fetch from
        limit_per_subreddit: Maximum posts per subreddit (default: 100)
        include_stickied: Whether to include pinned posts (default: False)
        
    Returns:
        Dict containing nested JSON structure with subreddits, posts, comments, and metadata
    """
    if not reddit:
        logging.error("PRAW Reddit instance not initialized for fetch_subreddit_posts_with_comments.")
        return {"subreddits": {}, "metadata": {"fetch_timestamp": datetime.now(timezone.utc).isoformat(), "total_subreddits": 0, "total_posts": 0, "total_comments": 0}}
    
    if not subreddit_names:
        logging.warning("No subreddit names provided to fetch_subreddit_posts_with_comments.")
        return {"subreddits": {}, "metadata": {"fetch_timestamp": datetime.now(timezone.utc).isoformat(), "total_subreddits": 0, "total_posts": 0, "total_comments": 0}}
    
    fetch_timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "subreddits": {},
        "metadata": {
            "fetch_timestamp": fetch_timestamp,
            "total_subreddits": 0,
            "total_posts": 0,
            "total_comments": 0
        }
    }
    
    total_posts = 0
    total_comments = 0
    successful_subreddits = 0
    
    logging.info(f"Starting fetch from {len(subreddit_names)} subreddits with limit {limit_per_subreddit} per subreddit")
    
    for subreddit_name in subreddit_names:
        logging.info(f"Processing subreddit: r/{subreddit_name}")
        
        try:
            subreddit_instance = reddit.subreddit(subreddit_name)
            
            # Fetch new submissions using existing helper function
            submissions_list = get_new_submissions_with_retry(subreddit_instance, limit=limit_per_subreddit)
            
            if not submissions_list:
                logging.info(f"No submissions found in r/{subreddit_name}")
                result["subreddits"][subreddit_name] = {"posts": []}
                continue
            
            subreddit_posts = []
            subreddit_comments_count = 0
            
            for submission in submissions_list:
                # Skip stickied posts if not included
                if not include_stickied and submission.stickied:
                    continue
                
                # Create post data structure
                post_data = {
                    "post_id": f"reddit-{submission.id}",
                    "title": submission.title,
                    "body": submission.selftext,
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "created_utc": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "upvote_ratio": getattr(submission, 'upvote_ratio', None),
                    "url": submission.url,
                    "subreddit": submission.subreddit.display_name,
                    "is_stickied": submission.stickied,
                    "comments": []
                }

                # Add image URLs (if present) using PRAW fields
                try:
                    image_urls = extract_image_urls_from_submission(submission)
                except Exception as _img_err:
                    logging.debug(f"Could not extract image URLs for post {submission.id}: {_img_err}")
                    image_urls = []
                post_data["image_urls"] = image_urls
                
                # Fetch comments using existing helper function
                try:
                    comment_list = get_comments_with_retry(submission)
                    
                    for comment in comment_list:
                        # Skip deleted/removed comments
                        if hasattr(comment, 'body') and comment.body not in ['[deleted]', '[removed]']:
                            comment_data = {
                                "comment_id": f"reddit-{comment.id}",
                                "body": comment.body,
                                "author": str(comment.author) if comment.author else "[deleted]",
                                "created_utc": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(),
                                "score": comment.score,
                                "parent_post_id": f"reddit-{submission.id}",
                                "permalink": f"https://reddit.com{comment.permalink}"
                            }
                            post_data["comments"].append(comment_data)
                            subreddit_comments_count += 1
                    
                    logging.debug(f"Fetched {len(post_data['comments'])} comments for post {submission.id}")
                    
                except Exception as e_comment:
                    logging.warning(f"Could not fetch comments for post {submission.id} in r/{subreddit_name}: {e_comment}")
                    # Continue without comments for this post
                
                subreddit_posts.append(post_data)
            
            # Add subreddit data to result
            result["subreddits"][subreddit_name] = {"posts": subreddit_posts}
            
            posts_count = len(subreddit_posts)
            total_posts += posts_count
            total_comments += subreddit_comments_count
            successful_subreddits += 1
            
            logging.info(f"Successfully processed r/{subreddit_name}: {posts_count} posts, {subreddit_comments_count} comments")
            
        except (prawcore.exceptions.Redirect, prawcore.exceptions.NotFound):
            logging.warning(f"Subreddit r/{subreddit_name} not found or redirected - skipping")
            continue
            
        except prawcore.exceptions.Forbidden:
            logging.warning(f"Cannot access private subreddit r/{subreddit_name} - skipping")
            continue
            
        except Exception as e_sub:
            logging.error(f"Unexpected error processing r/{subreddit_name}: {e_sub} - skipping")
            continue
    
    # Update metadata
    result["metadata"]["total_subreddits"] = successful_subreddits
    result["metadata"]["total_posts"] = total_posts
    result["metadata"]["total_comments"] = total_comments
    
    logging.info(f"Fetch completed: {successful_subreddits}/{len(subreddit_names)} subreddits, {total_posts} posts, {total_comments} comments")
    
    return result



def fetch_subreddit_posts_with_comments_time_windowed(
    reddit: praw.Reddit,
    subreddit_names: List[str],
    target_posts_per_subreddit: int = 5000,
    window_hours: int = 24,
    include_stickied: bool = False,
    fetch_comments: bool = False
) -> Dict[str, Any]:
    """Fetch posts while walking backwards through Reddit history.

    The previous implementation attempted to walk fixed-width time windows
    (e.g. 24 hours at a time) and would stop after a long "empty window" streak.
    That meant quiet subreddits – including r/HVAC – could hit the empty-window
    limit before ever encountering a real post, which is exactly what the logs
    in the bug report show.  We simplify the approach by repeatedly asking for
    "the newest posts before a moving timestamp".  Each batch advances the
    timestamp to the oldest item returned, so we always make progress even if a
    subreddit has weeks or months without activity.
    """

    # ``window_hours`` is retained for backwards compatibility with previous
    # callers.  The new implementation does not rely on fixed-width windows, so
    # the argument is currently unused.

    if not reddit:
        logging.error("PRAW Reddit instance not initialized for time-windowed fetch.")
        return {
            "subreddits": {},
            "metadata": {
                "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_subreddits": 0,
                "total_posts": 0,
                "total_comments": 0,
            },
        }

    if not subreddit_names:
        logging.warning("No subreddit names provided to time-windowed fetch.")
        return {
            "subreddits": {},
            "metadata": {
                "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                "total_subreddits": 0,
                "total_posts": 0,
                "total_comments": 0,
            },
        }

    fetch_timestamp = datetime.now(timezone.utc).isoformat()
    result: Dict[str, Any] = {
        "subreddits": {},
        "metadata": {
            "fetch_timestamp": fetch_timestamp,
            "total_subreddits": 0,
            "total_posts": 0,
            "total_comments": 0,
        },
    }

    total_posts = 0
    total_comments = 0
    successful_subreddits = 0

    logging.info(
        "Starting time-windowed fetch from %d subreddits; target %d posts/subreddit, fetch_comments=%s",
        len(subreddit_names),
        target_posts_per_subreddit,
        fetch_comments,
    )

    # Reddit search endpoints cap `limit` at ~100.  Keep batches small so we
    # can steadily move the `before` cursor backwards without hammering the API.
    search_batch_limit = 100

    for subreddit_name in subreddit_names:
        logging.info("[time-windowed] Processing subreddit: r/%s", subreddit_name)
        try:
            subreddit_instance = reddit.subreddit(subreddit_name)
            subreddit_posts: List[Dict[str, Any]] = []
            subreddit_comments_count = 0
            seen_ids: set[str] = set()

            before_timestamp = int(datetime.now(timezone.utc).timestamp())

            while len(subreddit_posts) < target_posts_per_subreddit and before_timestamp > 0:
                remaining_slots = target_posts_per_subreddit - len(subreddit_posts)
                batch_limit = min(search_batch_limit, remaining_slots)

                # CloudSearch supports range queries.  Asking for everything with
                # a timestamp less than `before_timestamp` ensures we keep
                # discovering older content without wasting calls on empty
                # windows.
                query = f"timestamp:<{before_timestamp}"
                submissions = search_submissions_with_retry(
                    subreddit_instance,
                    query=query,
                    sort="new",
                    syntax="cloudsearch",
                    limit=batch_limit,
                )

                if not submissions:
                    logging.debug(
                        "[time-windowed] No submissions for r/%s before %s; stopping",
                        subreddit_name,
                        datetime.fromtimestamp(before_timestamp, tz=timezone.utc).isoformat(),
                    )
                    break

                submissions.sort(
                    key=lambda sub: getattr(sub, "created_utc", 0), reverse=True
                )

                oldest_seen_ts: Optional[int] = None

                for submission in submissions:
                    created_utc = int(getattr(submission, "created_utc", 0))
                    oldest_seen_ts = (
                        created_utc
                        if oldest_seen_ts is None
                        else min(oldest_seen_ts, created_utc)
                    )

                    if not include_stickied and getattr(submission, "stickied", False):
                        continue
                    if submission.id in seen_ids:
                        continue

                    seen_ids.add(submission.id)

                    post_created = datetime.fromtimestamp(
                        created_utc, tz=timezone.utc
                    ).isoformat()

                    post_data: Dict[str, Any] = {
                        "post_id": f"reddit-{submission.id}",
                        "title": getattr(submission, "title", None),
                        "body": getattr(submission, "selftext", None),
                        "author": str(submission.author) if getattr(submission, "author", None) else "[deleted]",
                        "created_utc": post_created,
                        "score": getattr(submission, "score", None),
                        "num_comments": getattr(submission, "num_comments", None),
                        "upvote_ratio": getattr(submission, "upvote_ratio", None),
                        "url": getattr(submission, "url", None),
                        "subreddit": subreddit_name,
                        "is_stickied": getattr(submission, "stickied", False),
                        "comments": [],
                    }

                    try:
                        post_data["image_urls"] = extract_image_urls_from_submission(submission)
                    except Exception as err:
                        logging.debug(
                            "Could not extract image URLs for post %s: %s", submission.id, err
                        )
                        post_data["image_urls"] = []

                    if fetch_comments:
                        try:
                            for comment in get_comments_with_retry(submission):
                                if getattr(comment, "body", None) in ("[deleted]", "[removed]", None):
                                    continue
                                comment_data = {
                                    "comment_id": f"reddit-{comment.id}",
                                    "body": comment.body,
                                    "author": str(comment.author) if comment.author else "[deleted]",
                                    "created_utc": datetime.fromtimestamp(
                                        comment.created_utc, tz=timezone.utc
                                    ).isoformat(),
                                    "score": comment.score,
                                    "parent_post_id": f"reddit-{submission.id}",
                                    "permalink": f"https://reddit.com{comment.permalink}",
                                }
                                post_data["comments"].append(comment_data)
                                subreddit_comments_count += 1
                        except Exception as comment_err:
                            logging.warning(
                                "Could not fetch comments for post %s in r/%s: %s",
                                submission.id,
                                subreddit_name,
                                comment_err,
                            )

                    subreddit_posts.append(post_data)

                    if len(subreddit_posts) >= target_posts_per_subreddit:
                        break

                if oldest_seen_ts is None:
                    logging.debug(
                        "[time-windowed] Unable to determine oldest timestamp for r/%s; stopping",
                        subreddit_name,
                    )
                    break

                before_timestamp = max(0, oldest_seen_ts - 1)

            result["subreddits"][subreddit_name] = {"posts": subreddit_posts}
            posts_count = len(subreddit_posts)
            total_posts += posts_count
            total_comments += subreddit_comments_count
            successful_subreddits += 1
            logging.info(
                "[time-windowed] Completed r/%s: %d posts, %d comments",
                subreddit_name,
                posts_count,
                subreddit_comments_count,
            )

        except (prawcore.exceptions.Redirect, prawcore.exceptions.NotFound):
            logging.warning(
                "[time-windowed] Subreddit r/%s not found or redirected - skipping",
                subreddit_name,
            )
            continue
        except prawcore.exceptions.Forbidden:
            logging.warning(
                "[time-windowed] Cannot access private subreddit r/%s - skipping",
                subreddit_name,
            )
            continue
        except Exception as err:
            logging.error(
                "[time-windowed] Unexpected error processing r/%s: %s - skipping",
                subreddit_name,
                err,
            )
            continue

    result["metadata"]["total_subreddits"] = successful_subreddits
    result["metadata"]["total_posts"] = total_posts
    result["metadata"]["total_comments"] = total_comments
    logging.info(
        "[time-windowed] Fetch completed: %d/%d subreddits, %d posts, %d comments",
        successful_subreddits,
        len(subreddit_names),
        total_posts,
        total_comments,
    )
    return result


def HVAC_fetch(limit_per_subreddit: int = 100, include_stickied: bool = False):
    """Test that the fetch_subreddit_posts_with_comments function works correctly"""
    print("=== Testing fetch_subreddit_posts_with_comments function ===\n")
    
    # Set logging to DEBUG to see detailed progress
    logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Import Reddit credentials from local_secrets
        from local_secrets import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
        
        # Initialize Reddit client
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET, 
            user_agent=REDDIT_USER_AGENT,
            read_only=True
        )
        
        # Test connection
        reddit.user.me()
        print("✅ Reddit client initialized and connected successfully")
        
        # Test with a larger subreddit for more posts
        print("Fetching posts from r/AskReddit (limit=500)...")
        result = fetch_subreddit_posts_with_comments(
            reddit=reddit,
            subreddit_names=["HVAC"],
            limit_per_subreddit=limit_per_subreddit,
            include_stickied=include_stickied
        )
        
        print("📋 Result structure:")
        print(json.dumps(result, indent=2))
        
        # Verify structure
        assert "subreddits" in result, "Missing 'subreddits' key"
        assert "metadata" in result, "Missing 'metadata' key"
        
        # Check that we have the expected subreddits (some may be empty)
        # expected_subreddits = ["skilledtrades", "electricians", "Plumbing", "HVAC"]
        expected_subreddits = ["HVAC"]

        for subreddit_name in expected_subreddits:
            if subreddit_name in result["subreddits"]:
                assert "posts" in result["subreddits"][subreddit_name], f"Missing 'posts' in {subreddit_name}"
        
        # Calculate totals from all subreddits
        posts_count = result["metadata"]["total_posts"]
        total_comments = result["metadata"]["total_comments"]
        
        # Save to JSON file in output folder
        from datetime import date
        import os
        
        # Ensure output directory exists
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with current date
        today = date.today().strftime("%Y-%m-%d")
        filename = f"reddit_research_data_{today}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Write JSON to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Test passed - Fetched {posts_count} posts with {total_comments} total comments")
        print(f"📊 Metadata: {result['metadata']}")
        print(f"💾 Data saved to: {filepath}")
        
        return True
        
    except ImportError:
        print("❌ Test failed - local_secrets.py not found")
        print("     Create local_secrets.py with REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def HVAC_fetch_time_windowed(
    target_posts_per_subreddit: int = 5000,
    window_hours: int = 24,
    include_stickied: bool = False,
    fetch_comments: bool = False
):
    """
    Test runner for the time-windowed fetch to gather large volumes of posts.
    """
    print("=== Testing fetch_subreddit_posts_with_comments_time_windowed function ===\n")
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        from local_secrets import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET, 
            user_agent=REDDIT_USER_AGENT,
            read_only=True
        )
        reddit.user.me()
        print("✅ Reddit client initialized and connected successfully")
        
        result = fetch_subreddit_posts_with_comments_time_windowed(
            reddit=reddit,
            subreddit_names=["HVAC"],
            target_posts_per_subreddit=target_posts_per_subreddit,
            window_hours=window_hours,
            include_stickied=include_stickied,
            fetch_comments=fetch_comments
        )
        
        print("📋 Result structure:")
        print(json.dumps(result, indent=2))
        
        posts_count = result["metadata"]["total_posts"]
        total_comments = result["metadata"]["total_comments"]
        
        from datetime import date
        import os
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        today = date.today().strftime("%Y-%m-%d")
        filename = f"reddit_research_data_timewindowed_{today}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Time-windowed fetch - Fetched {posts_count} posts with {total_comments} total comments")
        print(f"📊 Metadata: {result['metadata']}")
        print(f"💾 Data saved to: {filepath}")
        return True
    except ImportError:
        print("❌ Test failed - local_secrets.py not found")
        print("     Create local_secrets.py with REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing HVAC fetch...")
    HVAC_fetch(limit_per_subreddit=5000, include_stickied=False)
    print("\nHVAC fetch complete!")
    print("\nTesting HVAC time-windowed fetch for 5000 posts...")
    HVAC_fetch_time_windowed(target_posts_per_subreddit=5000, window_hours=24, include_stickied=False, fetch_comments=False)
    print("\nHVAC time-windowed fetch complete!")

