import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import Post, Comment, SQLModel, Annotation
from sqlmodel import Session, create_engine, select
from scraper import is_low_quality, translate_query

# Create an in-memory engine for testing
test_engine = create_engine("sqlite:///:memory:")
SQLModel.metadata.create_all(test_engine)

@pytest.fixture(autouse=True)
def override_engine():
    # Clear and recreate database state for each test
    SQLModel.metadata.drop_all(test_engine)
    SQLModel.metadata.create_all(test_engine)
    
    # Patch the engine in all modules that use it
    with patch("database.engine", test_engine), \
         patch("scraper.engine", test_engine), \
         patch("analyzer.engine", test_engine), \
         patch("util.engine", test_engine):
        yield

def test_translate_query():
    assert translate_query("{Jake AND Amy}") == "(Jake AND Amy)"
    assert translate_query("Jake") == "Jake"

def test_is_low_quality():
    assert is_low_quality("AutoModerator", "some text") == True
    assert is_low_quality("RealUser", "[deleted]") == True
    assert is_low_quality("RealUser", "[removed]") == True
    assert is_low_quality("RealUser", "Good text") == False

def test_stage_2_label_cleanup_rejects_corrupt_cluster_label():
    from analyzer import clean_label_text, fallback_cluster_label, is_generic_cluster_label

    assert clean_label_text("Cluster 12\x00\x00") == "Cluster 12"
    assert is_generic_cluster_label("Cluster 12\x00\x00") is True

    annotations = [
        Annotation(
            item_id="ann_cleanup_1",
            item_type="post",
            sentiment=0.5,
            summary="Viewers liked the gradual romantic buildup.",
            raw_tag="earned romantic buildup",
        )
    ]
    assert fallback_cluster_label(annotations) == "earned romantic buildup"

def test_openrouter_retry_helper_retries_rate_limit_errors():
    from analyzer import run_openrouter_with_retries

    calls = {"count": 0}

    def flaky_operation():
        calls["count"] += 1
        if calls["count"] == 1:
            raise Exception("429 rate limit")
        return "ok"

    with patch("analyzer.time.sleep") as mock_sleep:
        assert run_openrouter_with_retries(flaky_operation, "test request") == "ok"

    assert calls["count"] == 2
    mock_sleep.assert_called_once()

@patch("scraper.get_praw_reddit")
def test_scrape_reddit_mock(mock_get_reddit):
    # Mock PRAW
    mock_reddit = MagicMock()
    mock_get_reddit.return_value = mock_reddit
    
    mock_sub = MagicMock()
    mock_reddit.subreddit.return_value = mock_sub
    
    mock_submission = MagicMock()
    mock_submission.id = "post1"
    mock_submission.subreddit.display_name = "testsub"
    mock_submission.title = "Test Post"
    mock_submission.selftext = "Test Body"
    mock_submission.score = 10
    mock_submission.created_utc = 1000000.0
    mock_submission.author.name = "User"
    
    mock_comment = MagicMock()
    mock_comment.id = "comment1"
    mock_comment.body = "Test Comment"
    mock_comment.score = 5
    mock_comment.created_utc = 1000010.0
    mock_comment.author.name = "User2"
    
    comments_mock = MagicMock()
    comments_mock.replace_more = MagicMock()
    comments_mock.__iter__.return_value = iter([mock_comment])
    mock_submission.comments = comments_mock
    
    mock_sub.search.return_value = [mock_submission]
    
    from scraper import scrape_reddit
    # We do a tiny scrape
    scrape_reddit("query", subreddits=["testsub"], post_limit=1, comment_limit=1)
    assert mock_submission.comment_sort == "confidence"
    
    # Check DB
    with Session(test_engine) as session:
        post = session.get(Post, "post1")
        assert post is not None
        assert post.status == "pending"
        
        comment = session.get(Comment, "comment1")
        assert comment is not None
        assert comment.status == "pending"

@patch("analyzer.OpenRouter")
@patch.dict('os.environ', {"OPENROUTER_API_KEY": "test_key"})
def test_stage_1_analysis(mock_openrouter):
    # Mock OpenRouter Response
    mock_client_instance = MagicMock()
    mock_openrouter.return_value.__enter__.return_value = mock_client_instance
    
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"sentiment": 0.5, "summary": "A good post.", "raw_tag": "positive feedback"}'
    mock_response.choices = [mock_choice]
    mock_client_instance.chat.send.return_value = mock_response
    
    with Session(test_engine) as session:
        # Add a pending post manually
        p = Post(id="post_test_1", subreddit="test", title="T", selftext="S", score=1, created_utc=0.0)
        session.add(p)
        session.commit()
    
    from analyzer import run_stage_1_analysis
    run_stage_1_analysis()
    
    with Session(test_engine) as session:
        post = session.get(Post, "post_test_1")
        assert post.status == "processed"
        
        annotation = session.get(Annotation, "post_test_1")
        assert annotation is not None
        assert annotation.item_type == "post"
        assert annotation.sentiment == 0.5
        assert annotation.raw_tag == "positive feedback"
        assert annotation.consolidated_tag is None
        assert annotation.cluster_id is None
        assert annotation.embedding is None

@patch("analyzer.HDBSCAN")
@patch("analyzer.OpenRouter")
@patch.dict('os.environ', {"OPENROUTER_API_KEY": "test_key"})
def test_stage_2_analysis(mock_openrouter, mock_hdbscan):
    # Mock HDBSCAN
    mock_clusterer = MagicMock()
    mock_clusterer.fit_predict.return_value = np.array([0, 0])
    mock_hdbscan.return_value = mock_clusterer

    # Mock OpenRouter Response (both embeddings and chat labels)
    mock_client_instance = MagicMock()
    mock_openrouter.return_value.__enter__.return_value = mock_client_instance
    
    # Mock Embeddings return value
    mock_emb_res = MagicMock()
    mock_emb_data = MagicMock()
    mock_emb_data.embedding = [0.1] * 384
    mock_emb_res.data = [mock_emb_data]
    mock_client_instance.embeddings.generate.return_value = mock_emb_res
    
    # Mock Chat Label return value
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"consolidated_tag": "Appreciation", "explanation": "These reactions share appreciative responses to the relationship dynamic."}'
    mock_response.choices = [mock_choice]
    mock_client_instance.chat.send.return_value = mock_response
    
    with Session(test_engine) as session:
        # Add two unmapped annotations to allow min_cluster_size=2
        ann1 = Annotation(
            item_id="post_test_2",
            item_type="post",
            sentiment=0.5,
            summary="A good post.",
            raw_tag="positive feedback",
            consolidated_tag=None
        )
        ann2 = Annotation(
            item_id="post_test_3",
            item_type="post",
            sentiment=0.5,
            summary="Another good post.",
            raw_tag="positive feedback",
            consolidated_tag=None
        )
        session.add(ann1)
        session.add(ann2)
        session.commit()
        
    from analyzer import run_stage_2_analysis
    run_stage_2_analysis(min_cluster_size=2, labeling_model="test-model")
    
    with Session(test_engine) as session:
        annotation = session.get(Annotation, "post_test_2")
        assert annotation.consolidated_tag == "Appreciation"
        assert annotation.cluster_explanation == "These reactions share appreciative responses to the relationship dynamic."
        assert annotation.cluster_id == 0
        assert annotation.embedding is not None

@patch("visualization.handle_output")
def test_visualizations(mock_handle_output):
    # Setup mock posts, comments, and annotations in test DB
    import json
    with Session(test_engine) as session:
        p1 = Post(id="post_v1", subreddit="r/romance", title="Post V1", selftext="SelfText", score=10, created_utc=1000000.0)
        c1 = Comment(id="comment_v1", post_id="post_v1", body="Comment V1", score=5, created_utc=1000060.0)
        
        ann_p = Annotation(
            item_id="post_v1",
            item_type="post",
            sentiment=0.5,
            summary="Summary Post",
            raw_tag="love",
            consolidated_tag="Affection",
            cluster_id=0,
            embedding=json.dumps([0.1] * 384)
        )
        
        ann_c = Annotation(
            item_id="comment_v1",
            item_type="comment",
            sentiment=-0.5,
            summary="Summary Comment",
            raw_tag="sad",
            consolidated_tag="Angst",
            cluster_id=1,
            embedding=json.dumps([-0.1] * 384)
        )
        
        session.add(p1)
        session.add(c1)
        session.add(ann_p)
        session.add(ann_c)
        session.commit()

    from visualization import (
        plot_semantic_map, plot_sentiment_distribution, plot_sentiment_by_theme,
        plot_theme_dominance_bar, plot_theme_dominance_pareto
    )
    
    with Session(test_engine) as session:
        # Run all visualization functions (interactive / preview mode)
        plot_semantic_map(session)
        plot_sentiment_distribution(session)
        plot_sentiment_by_theme(session)
        plot_theme_dominance_bar(session)
        plot_theme_dominance_pareto(session)
        
        # Run in export mode
        plot_semantic_map(session, save_path="plots/test_map.png")
        plot_sentiment_distribution(session, save_path="plots/test_sentiment_distribution.png")
        plot_sentiment_by_theme(session, save_path="plots/test_sentiment.png")
        plot_theme_dominance_bar(session, save_path="plots/test_dominance_bar.png")
        plot_theme_dominance_pareto(session, save_path="plots/test_dominance_pareto.png")

    assert mock_handle_output.call_count == 10

@patch("scraper.fetch_json")
@patch("scraper.PlaywrightManager")
@patch("scraper.fetch_json_playwright")
def test_scrape_reddit_json_mock(mock_fetch_json_playwright, mock_playwright_manager, mock_fetch_json):
    # Mock Playwright to fail
    mock_fetch_json_playwright.side_effect = Exception("Playwright failed")
    
    # Mock the return values for search and comments matching PullPush structure
    mock_search_res = {
        "data": [
            {
                "id": "post_json1",
                "subreddit": "testsub",
                "title": "Test Post JSON",
                "selftext": "Test Body JSON",
                "score": 10,
                "created_utc": 1000000.0,
                "author": "User"
            }
        ]
    }

    mock_comments_res = {
        "data": [
            {
                "id": "comment_json1",
                "body": "Test Comment JSON",
                "score": 5,
                "created_utc": 1000010.0,
                "author": "User2"
            }
        ]
    }

    # Simulate:
    # 1. Direct search fails -> Playwright search fails -> PullPush search succeeds
    # 2. Direct comment fails -> Playwright comment fails -> PullPush comment succeeds
    mock_fetch_json.side_effect = [
        Exception("Direct search 403 Forbidden"),
        mock_search_res,
        Exception("Direct comments 403 Forbidden"),
        mock_comments_res
    ]

    from scraper import scrape_reddit
    with patch("time.sleep"):
        scrape_reddit("query", subreddits=["testsub"], post_limit=1, comment_limit=1, method="json")

    with Session(test_engine) as session:
        post = session.get(Post, "post_json1")
        assert post is not None
        assert post.title == "Test Post JSON"

        comment = session.get(Comment, "comment_json1")
        assert comment is not None
        assert comment.body == "Test Comment JSON"

    pullpush_comment_urls = [call.args[0] for call in mock_fetch_json.call_args_list if "search/comment" in call.args[0]]
    assert pullpush_comment_urls
    assert "sort_type=score" in pullpush_comment_urls[0]
    assert "sort=desc" in pullpush_comment_urls[0]


@patch("scraper.fetch_json")
@patch("scraper.PlaywrightManager")
@patch("scraper.fetch_json_playwright")
def test_scrape_reddit_json_direct_success_mock(mock_fetch_json_playwright, mock_playwright_manager, mock_fetch_json):
    # Mock the return values for search and comments matching Reddit's structure
    mock_search_res = {
        "data": {
            "children": [
                {
                    "data": {
                        "id": "direct_post1",
                        "subreddit": "testsub",
                        "title": "Direct Post Title",
                        "selftext": "Direct Post Body",
                        "score": 15,
                        "created_utc": 2000000.0,
                        "author": "DirectUser"
                    }
                }
            ]
        }
    }
    
    mock_comments_res = [
        {}, # first element (post details)
        {   # second element (comments list)
            "data": {
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "id": "direct_comment1",
                            "body": "Direct Comment Body",
                            "score": 8,
                            "created_utc": 2000010.0,
                            "author": "DirectUser2"
                        }
                    }
                ]
            }
        }
    ]
    
    mock_fetch_json.side_effect = [mock_search_res, mock_comments_res]
    
    from scraper import scrape_reddit
    with patch("time.sleep"):
        scrape_reddit("query", subreddits=["testsub"], post_limit=1, comment_limit=1, method="json")
        
    with Session(test_engine) as session:
        post = session.get(Post, "direct_post1")
        assert post is not None
        assert post.title == "Direct Post Title"
        
        comment = session.get(Comment, "direct_comment1")
        assert comment is not None
        assert comment.body == "Direct Comment Body"

    comment_urls = [call.args[0] for call in mock_fetch_json.call_args_list if "/comments/" in call.args[0]]
    assert comment_urls
    assert "sort=confidence" in comment_urls[0]


@patch("scraper.fetch_json")
@patch("scraper.PlaywrightManager")
@patch("scraper.fetch_json_playwright")
def test_scrape_reddit_json_fill_post_limit_fetches_past_skipped(mock_fetch_json_playwright, mock_playwright_manager, mock_fetch_json):
    first_search_res = {
        "data": {
            "after": "t3_after_first",
            "children": [
                {
                    "data": {
                        "id": "skipped_post",
                        "subreddit": "testsub",
                        "title": "Skipped Post",
                        "selftext": "",
                        "score": 1,
                        "created_utc": 100.0,
                        "author": "RealUser"
                    }
                }
            ]
        }
    }

    second_search_res = {
        "data": {
            "after": None,
            "children": [
                {
                    "data": {
                        "id": "usable_post",
                        "subreddit": "testsub",
                        "title": "Usable Post",
                        "selftext": "This post has body text.",
                        "score": 2,
                        "created_utc": 200.0,
                        "author": "RealUser"
                    }
                }
            ]
        }
    }

    empty_comments_res = [{}, {"data": {"children": []}}]
    mock_fetch_json.side_effect = [
        first_search_res,
        empty_comments_res,
        second_search_res,
        empty_comments_res,
    ]

    from scraper import scrape_reddit
    with patch("time.sleep"):
        scrape_reddit(
            "query",
            subreddits=["testsub"],
            post_limit=1,
            comment_limit=1,
            method="json",
            fill_post_limit=True,
        )

    with Session(test_engine) as session:
        skipped_post = session.get(Post, "skipped_post")
        usable_post = session.get(Post, "usable_post")
        pending_posts = session.exec(select(Post).where(Post.status == "pending")).all()

        assert skipped_post is not None
        assert skipped_post.status == "skipped"
        assert usable_post is not None
        assert usable_post.status == "pending"
        assert len(pending_posts) == 1

    search_urls = [call.args[0] for call in mock_fetch_json.call_args_list if "search.json" in call.args[0]]
    assert len(search_urls) == 2
    assert "after=t3_after_first" in search_urls[1]


@patch("scraper.fetch_json")
@patch("scraper.PlaywrightManager")
@patch("scraper.fetch_json_playwright")
def test_scrape_reddit_json_playwright_success_mock(mock_fetch_json_playwright, mock_playwright_manager, mock_fetch_json):
    # Mock direct search/comments to fail
    mock_fetch_json.side_effect = Exception("Direct fetch 403 Forbidden")
    
    # Mock Playwright search and comments to succeed
    mock_search_res = {
        "data": {
            "children": [
                {
                    "data": {
                        "id": "pw_post1",
                        "subreddit": "testsub",
                        "title": "Playwright Post Title",
                        "selftext": "Playwright Post Body",
                        "score": 20,
                        "created_utc": 3000000.0,
                        "author": "PWUser"
                    }
                }
            ]
        }
    }
    
    mock_comments_res = [
        {}, # first element (post details)
        {   # second element (comments list)
            "data": {
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "id": "pw_comment1",
                            "body": "Playwright Comment Body",
                            "score": 12,
                            "created_utc": 3000010.0,
                            "author": "PWUser2"
                        }
                    }
                ]
            }
        }
    ]
    
    mock_fetch_json_playwright.side_effect = [mock_search_res, mock_comments_res]
    
    from scraper import scrape_reddit
    with patch("time.sleep"):
        scrape_reddit("query", subreddits=["testsub"], post_limit=1, comment_limit=1, method="json")
        
    with Session(test_engine) as session:
        post = session.get(Post, "pw_post1")
        assert post is not None
        assert post.title == "Playwright Post Title"
        
        comment = session.get(Comment, "pw_comment1")
        assert comment is not None
        assert comment.body == "Playwright Comment Body"


@patch("scraper.fetch_json")
@patch("scraper.PlaywrightManager")
@patch("scraper.fetch_json_playwright")
def test_scrape_reddit_playwright_direct_mock(mock_fetch_json_playwright, mock_playwright_manager, mock_fetch_json):
    # Mock Playwright search and comments to succeed
    mock_search_res = {
        "data": {
            "children": [
                {
                    "data": {
                        "id": "pw_direct_post1",
                        "subreddit": "testsub",
                        "title": "Playwright Direct Title",
                        "selftext": "Playwright Direct Body",
                        "score": 30,
                        "created_utc": 4000000.0,
                        "author": "PWDirectUser"
                    }
                }
            ]
        }
    }
    
    mock_comments_res = [
        {},
        {
            "data": {
                "children": [
                    {
                        "kind": "t1",
                        "data": {
                            "id": "pw_direct_comment1",
                            "body": "Playwright Direct Comment",
                            "score": 15,
                            "created_utc": 4000010.0,
                            "author": "PWDirectUser2"
                        }
                    }
                ]
            }
        }
    ]
    
    mock_fetch_json_playwright.side_effect = [mock_search_res, mock_comments_res]
    
    from scraper import scrape_reddit
    with patch("time.sleep"):
        scrape_reddit("query", subreddits=["testsub"], post_limit=1, comment_limit=1, method="playwright")
        
    with Session(test_engine) as session:
        post = session.get(Post, "pw_direct_post1")
        assert post is not None
        assert post.title == "Playwright Direct Title"
        
        comment = session.get(Comment, "pw_direct_comment1")
        assert comment is not None
        assert comment.body == "Playwright Direct Comment"

    comment_urls = [call.args[1] for call in mock_fetch_json_playwright.call_args_list if "/comments/" in call.args[1]]
    assert comment_urls
    assert "sort=confidence" in comment_urls[0]


