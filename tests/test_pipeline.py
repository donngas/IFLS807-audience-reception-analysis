import pytest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import Post, Comment, TagMapping, SQLModel
from sqlmodel import Session, create_engine
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
    
    # Check DB
    with Session(test_engine) as session:
        post = session.get(Post, "post1")
        assert post is not None
        assert post.status == "pending"
        
        comment = session.get(Comment, "comment1")
        assert comment is not None
        assert comment.status == "pending"

@patch("analyzer.ollama.Client")
def test_stage_1_analysis(mock_ollama_client):
    # Mock Ollama Response
    mock_client_instance = MagicMock()
    mock_ollama_client.return_value = mock_client_instance
    
    mock_response = MagicMock()
    mock_response.message.content = '{"sentiment": 0.5, "summary": "A good post.", "raw_tag": "positive feedback"}'
    mock_client_instance.chat.return_value = mock_response
    
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
        assert post.sentiment == 0.5
        assert post.raw_tag == "positive feedback"
        
        tag = session.get(TagMapping, "positive feedback")
        assert tag is not None
        assert tag.consolidated_tag is None

@patch("analyzer.genai.Client")
@patch.dict('os.environ', {"GEMINI_API_KEY": "test_key"})
def test_stage_2_analysis(mock_genai_client):
    # Mock Gemini Response
    mock_client_instance = MagicMock()
    mock_genai_client.return_value = mock_client_instance
    
    mock_response = MagicMock()
    mock_response.text = '{"tag_mappings": {"positive feedback": "Appreciation"}}'
    mock_client_instance.models.generate_content.return_value = mock_response
    
    with Session(test_engine) as session:
        # Add an unmapped tag
        t = TagMapping(raw_tag="positive feedback")
        session.add(t)
        session.commit()
        
    from analyzer import run_stage_2_analysis
    run_stage_2_analysis()
    
    with Session(test_engine) as session:
        tag = session.get(TagMapping, "positive feedback")
        assert tag.consolidated_tag == "Appreciation"
