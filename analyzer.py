import os
from typing import Literal
from pydantic import BaseModel, Field
import ollama
from google import genai
from google.genai import types

from database import (
    Session, engine, get_pending_posts, get_pending_comments,
    upsert_post, upsert_comment, upsert_tag_mapping, get_unmapped_raw_tags
)

# Stage 1 Schema
class Stage1Output(BaseModel):
    sentiment: Literal[-1.0, -0.5, 0.0, 0.5, 1.0] = Field(
        description="Sentiment polarity score: strongly negative (-1.0) to strongly positive (1.0)"
    )
    summary: str = Field(
        description="A concise 1-2 sentence summary of the post or comment."
    )
    raw_tag: str = Field(
        description="A primary 1-3 word descriptor theme (e.g., 'chemistry', 'pacing issues'). Use 'Reaction Only' if no specific analytical theme."
    )

# Stage 2 Schema
class Stage2Output(BaseModel):
    tag_mappings: dict[str, str] = Field(
        description="Key-value mapping where key is the raw_tag and value is the consolidated_tag (one of 10-15 broader analytical categories)."
    )

def run_stage_1_analysis():
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model_name = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
    
    client = ollama.Client(host=ollama_host)
    
    system_prompt = (
        "You are an expert audience reception analyst. Analyze the following Reddit text and extract "
        "the sentiment, summary, and a primary raw tag. "
        "Guideline: If the text is shorter than 15 characters or lacks analytical substance, "
        "classify it with the tag 'Reaction Only'."
    )
    
    with Session(engine) as session:
        # Process Posts
        pending_posts = get_pending_posts(session, limit=100)
        for post in pending_posts:
            text_to_analyze = f"Title: {post.title}\n\nBody: {post.selftext}"
            try:
                response = client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text_to_analyze}
                    ],
                    format=Stage1Output.model_json_schema()
                )
                output = Stage1Output.model_validate_json(response.message.content)
                post.sentiment = output.sentiment
                post.summary = output.summary
                post.raw_tag = output.raw_tag
                post.status = "processed"
                upsert_tag_mapping(session, output.raw_tag)
            except Exception as e:
                print(f"Failed to process post {post.id}: {e}")
                post.status = "failed"
            upsert_post(session, post)
            
        # Process Comments
        pending_comments = get_pending_comments(session, limit=100)
        for comment in pending_comments:
            try:
                response = client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": comment.body}
                    ],
                    format=Stage1Output.model_json_schema()
                )
                output = Stage1Output.model_validate_json(response.message.content)
                comment.sentiment = output.sentiment
                comment.summary = output.summary
                comment.raw_tag = output.raw_tag
                comment.status = "processed"
                upsert_tag_mapping(session, output.raw_tag)
            except Exception as e:
                print(f"Failed to process comment {comment.id}: {e}")
                comment.status = "failed"
            upsert_comment(session, comment)

def run_stage_2_analysis():
    # Ensure GEMINI_API_KEY is set or the client will fail
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY environment variable is required for Stage 2 clustering.")
    
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3-flash")
    # if gemini-3-flash is not available, might fall back to gemini-2.5-flash but we use what the env says.
    client = genai.Client()
    
    with Session(engine) as session:
        unmapped_tags = get_unmapped_raw_tags(session)
        # unmapped_tags is a list of strings
        if not unmapped_tags:
            print("No unmapped tags found. Skipping Stage 2.")
            return
            
        print(f"Clustering {len(unmapped_tags)} unmapped raw tags using {gemini_model}...")
        
        prompt = (
            "You are an expert audience reception analyst. Please group the following raw tags into "
            "10-15 broader analytical categories. Return a JSON object where the keys are the exact raw tags "
            "provided and the values are the corresponding consolidated tags.\n\n"
            f"Raw Tags to cluster:\n{unmapped_tags}"
        )
        
        try:
            response = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Stage2Output,
                )
            )
            
            output = Stage2Output.model_validate_json(response.text)
            
            for raw, consolidated in output.tag_mappings.items():
                upsert_tag_mapping(session, raw, consolidated)
                
            print("Stage 2 thematic clustering completed successfully.")
        except Exception as e:
            print(f"Failed during Stage 2 clustering: {e}")
