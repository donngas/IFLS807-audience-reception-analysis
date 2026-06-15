import os
from typing import Literal
from pydantic import BaseModel, Field
from openrouter import OpenRouter

from database import (
    Session, engine, get_pending_posts, get_pending_comments,
    upsert_post, upsert_comment, get_unmapped_raw_tags,
    Annotation, upsert_annotation, update_consolidated_tags
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
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for Stage 1 analysis.")
    
    model_name = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    
    system_prompt = (
        "You are an expert audience reception analyst. Analyze the following Reddit text and extract "
        "the sentiment, summary, and a primary raw tag. "
        "Guideline: If the text is shorter than 15 characters or lacks analytical substance, "
        "classify it with the tag 'Reaction Only'."
    )
    
    with OpenRouter(api_key=openrouter_api_key) as client:
        with Session(engine) as session:
            # Process Posts
            pending_posts = get_pending_posts(session, limit=100)
            for post in pending_posts:
                text_to_analyze = f"Title: {post.title}\n\nBody: {post.selftext}"
                try:
                    response = client.chat.send(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": text_to_analyze}
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "Stage1Output",
                                "strict": True,
                                "schema": Stage1Output.model_json_schema()
                            }
                        }
                    )
                    output = Stage1Output.model_validate_json(response.choices[0].message.content)
                    annotation = Annotation(
                        item_id=post.id,
                        item_type="post",
                        sentiment=output.sentiment,
                        summary=output.summary,
                        raw_tag=output.raw_tag
                    )
                    upsert_annotation(session, annotation)
                    post.status = "processed"
                except Exception as e:
                    print(f"Failed to process post {post.id}: {e}")
                    post.status = "failed"
                upsert_post(session, post)
                
            # Process Comments
            pending_comments = get_pending_comments(session, limit=100)
            for comment in pending_comments:
                try:
                    response = client.chat.send(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": comment.body}
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "Stage1Output",
                                "strict": True,
                                "schema": Stage1Output.model_json_schema()
                            }
                        }
                    )
                    output = Stage1Output.model_validate_json(response.choices[0].message.content)
                    annotation = Annotation(
                        item_id=comment.id,
                        item_type="comment",
                        sentiment=output.sentiment,
                        summary=output.summary,
                        raw_tag=output.raw_tag
                    )
                    upsert_annotation(session, annotation)
                    comment.status = "processed"
                except Exception as e:
                    print(f"Failed to process comment {comment.id}: {e}")
                    comment.status = "failed"
                upsert_comment(session, comment)

def run_stage_2_analysis():
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for Stage 2 clustering.")
    
    model_name = os.environ.get("OPENROUTER_MODEL_STAGE2", os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash"))
    
    with Session(engine) as session:
        unmapped_tags = get_unmapped_raw_tags(session)
        if not unmapped_tags:
            print("No unmapped tags found. Skipping Stage 2.")
            return
            
        print(f"Clustering {len(unmapped_tags)} unmapped raw tags using {model_name}...")
        
        prompt = (
            "You are an expert audience reception analyst. Please group the following raw tags into "
            "10-15 broader analytical categories. Return a JSON object where the keys are the exact raw tags "
            "provided and the values are the corresponding consolidated tags.\n\n"
            f"Raw Tags to cluster:\n{unmapped_tags}"
        )
        
        try:
            with OpenRouter(api_key=openrouter_api_key) as client:
                response = client.chat.send(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "Stage2Output",
                            "strict": True,
                            "schema": Stage2Output.model_json_schema()
                        }
                    }
                )
            output = Stage2Output.model_validate_json(response.choices[0].message.content)
            
            update_consolidated_tags(session, output.tag_mappings)
                
            print("Stage 2 thematic clustering completed successfully.")
        except Exception as e:
            print(f"Failed during Stage 2 clustering: {e}")
