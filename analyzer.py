import os
import json
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from tqdm import tqdm
import numpy as np
from sklearn.cluster import HDBSCAN
from openrouter import OpenRouter
from sqlmodel import select

from database import (
    Session, engine, get_pending_posts, get_pending_comments,
    upsert_post, upsert_comment, get_unmapped_raw_tags,
    Annotation, upsert_annotation, update_consolidated_tags, Post, Comment
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

# Stage 2 Schema (Cluster Labeling from LLM)
class ClusterLabelOutput(BaseModel):
    consolidated_tag: str = Field(
        description="A concise 1-3 word descriptor category for this cluster of reactions (e.g., 'Character Chemistry', 'Dialogue Quality', etc.)."
    )

def run_stage_1_analysis(model_name: Optional[str] = None, limit: int = 100, temperature: float = 0.1, force_reanalyze: bool = False):
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for Stage 1 analysis.")
    
    if not model_name:
        model_name = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it")
    
    system_prompt = (
        "You are an expert audience reception analyst. Analyze the following Reddit text and extract "
        "the sentiment, summary, and a primary raw tag. "
        "Guideline: If the text is shorter than 15 characters or lacks analytical substance, "
        "classify it with the tag 'Reaction Only'."
    )
    
    with OpenRouter(api_key=openrouter_api_key) as client:
        with Session(engine) as session:
            # Process Posts
            if force_reanalyze:
                posts_stmt = select(Post).where((Post.status == "pending") | (Post.status == "processed")).limit(limit)
                posts_to_process = session.exec(posts_stmt).all()
            else:
                posts_to_process = get_pending_posts(session, limit=limit)
                
            print(f"Stage 1: Processing {len(posts_to_process)} posts...")
            for post in tqdm(posts_to_process, desc="Analyzing Posts"):
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
                        },
                        temperature=temperature
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
            if force_reanalyze:
                comments_stmt = select(Comment).where((Comment.status == "pending") | (Comment.status == "processed")).limit(limit)
                comments_to_process = session.exec(comments_stmt).all()
            else:
                comments_to_process = get_pending_comments(session, limit=limit)
                
            print(f"Stage 1: Processing {len(comments_to_process)} comments...")
            for comment in tqdm(comments_to_process, desc="Analyzing Comments"):
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
                        },
                        temperature=temperature
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
    print("Stage 1 feature extraction completed.")

def run_stage_2_analysis(
    embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
    labeling_model: str = "google/gemma-4-26b-a4b-it",
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    force_reembed: bool = False,
    force_recluster: bool = False,
    label_sample_size: int = 10
):
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for Stage 2 analysis.")
        
    with Session(engine) as session:
        # Retrieve all processed annotations
        annotations_stmt = select(Annotation)
        annotations = session.exec(annotations_stmt).all()
        
        if not annotations:
            print("No annotations found. Please run Stage 1 first.")
            return
            
        print(f"Stage 2: Found {len(annotations)} annotations in the database.")
        
        # 1. Generate Embeddings (checking cache)
        with OpenRouter(api_key=openrouter_api_key) as client:
            embeddings_needed = []
            for ann in annotations:
                if force_reembed or not ann.embedding:
                    embeddings_needed.append(ann)
            
            if embeddings_needed:
                print(f"Generating embeddings for {len(embeddings_needed)} annotations using {embedding_model}...")
                for ann in tqdm(embeddings_needed, desc="Generating Embeddings"):
                    combined_text = f"Tag: {ann.raw_tag} | Summary: {ann.summary}"
                    try:
                        res = client.embeddings.generate(
                            model=embedding_model,
                            input=combined_text
                        )
                        embedding_vector = res.data[0].embedding
                        ann.embedding = json.dumps(embedding_vector)
                        upsert_annotation(session, ann)
                    except Exception as e:
                        print(f"Failed to generate embedding for {ann.item_id}: {e}")
                
                # Reload annotations to ensure we have all embeddings
                annotations = session.exec(annotations_stmt).all()
        
        # Filter annotations that have embeddings
        valid_annotations = [ann for ann in annotations if ann.embedding is not None]
        if not valid_annotations:
            print("No valid embeddings found. Cannot proceed with clustering.")
            return
            
        # Parse embeddings into numpy array
        vectors = []
        for ann in valid_annotations:
            vectors.append(json.loads(ann.embedding))
        X = np.array(vectors) # Shape: (num_annotations, num_dimensions)
        
        # Normalize vectors for cosine similarity distance checks
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        X_norm = X / norms
        
        # Check if we should recluster
        already_clustered = all(ann.cluster_id is not None for ann in valid_annotations)
        if already_clustered and not force_recluster:
            print("Annotations are already clustered. Use force_recluster=True to re-run.")
            return
            
        # Adjust min_cluster_size if dataset is too small
        n_samples = len(valid_annotations)
        effective_min_cluster_size = min_cluster_size
        if n_samples < min_cluster_size:
            effective_min_cluster_size = max(2, n_samples // 2)
            print(f"Warning: Dataset size ({n_samples}) is smaller than min_cluster_size ({min_cluster_size}). "
                  f"Temporarily reducing min_cluster_size to {effective_min_cluster_size}.")
        
        if n_samples < 2:
            print("Too few annotations to run clustering (need at least 2).")
            # Assign them all to cluster 0 and label them manually/Reaction Only
            for ann in valid_annotations:
                ann.cluster_id = 0
                ann.consolidated_tag = "Reaction Only"
                upsert_annotation(session, ann)
            return

        # 2. Run HDBSCAN
        print(f"Clustering {n_samples} annotations using HDBSCAN (min_cluster_size={effective_min_cluster_size}, min_samples={min_samples})...")
        clusterer = HDBSCAN(min_cluster_size=effective_min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(X_norm)
        
        # Save labels to annotations
        for idx, ann in enumerate(valid_annotations):
            ann.cluster_id = int(labels[idx])
            upsert_annotation(session, ann)
            
        # 3. Resolve Outliers (-1) using Cosine Similarity to Cluster Centroids
        unique_labels = set(labels)
        valid_labels = [l for l in unique_labels if l >= 0]
        
        if len(valid_labels) > 0:
            # Calculate centroids for each cluster
            centroids = {}
            for label in valid_labels:
                cluster_indices = [i for i, l in enumerate(labels) if l == label]
                cluster_vectors = X_norm[cluster_indices]
                mean_vector = np.mean(cluster_vectors, axis=0)
                mean_norm = np.linalg.norm(mean_vector)
                centroids[label] = mean_vector / (mean_norm if mean_norm > 0 else 1.0)
                
            # Assign noise points to the closest cluster centroid
            noise_indices = [i for i, l in enumerate(labels) if l == -1]
            if noise_indices:
                print(f"Resolving {len(noise_indices)} outlier noise points to closest cluster centroids...")
                for idx in noise_indices:
                    ann = valid_annotations[idx]
                    vector = X_norm[idx]
                    
                    # Compute cosine similarities
                    best_label = -1
                    best_sim = -1.0
                    for label, centroid in centroids.items():
                        sim = float(np.dot(vector, centroid))
                        if sim > best_sim:
                            best_sim = sim
                            best_label = label
                    
                    ann.cluster_id = best_label
                    upsert_annotation(session, ann)
            
            # 4. LLM Cluster Labeling
            print("Generating thematic labels for clusters using LLM...")
            cluster_mappings = {}
            with OpenRouter(api_key=openrouter_api_key) as client:
                for label in tqdm(valid_labels, desc="Labeling Clusters"):
                    # Get all annotations currently assigned to this cluster (including resolved outliers)
                    cluster_ann_indices = [i for i, ann in enumerate(valid_annotations) if ann.cluster_id == label]
                    cluster_annotations = [valid_annotations[i] for i in cluster_ann_indices]
                    
                    # Calculate distances to centroid
                    centroid = centroids[label]
                    distances = []
                    for ann_idx in cluster_ann_indices:
                        vec = X_norm[ann_idx]
                        dist = 1.0 - np.dot(vec, centroid) # cosine distance
                        distances.append(dist)
                        
                    # Sort annotations by distance to centroid (ascending)
                    sorted_ann_with_dist = sorted(zip(cluster_annotations, distances), key=lambda item: item[1])
                    
                    # Take top k representative items
                    sample_size = min(label_sample_size, len(sorted_ann_with_dist))
                    representative_items = sorted_ann_with_dist[:sample_size]
                    
                    # Format prompt
                    sample_text = ""
                    for item, dist in representative_items:
                        sample_text += f"- Raw Tag: '{item.raw_tag}' | Summary: '{item.summary}'\n"
                        
                    prompt = (
                        "You are an expert audience reception analyst studying serialized romance in television.\n"
                        "Below is a list of representative raw tags and summaries from a cluster of audience reactions:\n\n"
                        f"{sample_text}\n"
                        "Please group these reactions under a single cohesive, consolidated tag. "
                        "The consolidated tag should be 1 to 3 words in length (e.g., 'Character Chemistry', 'Dialogue Quality', 'Pacing Issues').\n"
                        "Return a JSON object matching this schema:\n"
                        "{\n"
                        "  \"consolidated_tag\": \"your label\"\n"
                        "}"
                    )
                    
                    try:
                        response = client.chat.send(
                            model=labeling_model,
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "ClusterLabelOutput",
                                    "strict": True,
                                    "schema": ClusterLabelOutput.model_json_schema()
                                }
                            },
                            temperature=0.1
                        )
                        output = ClusterLabelOutput.model_validate_json(response.choices[0].message.content)
                        consolidated_tag = output.consolidated_tag.strip()
                        cluster_mappings[label] = consolidated_tag
                        print(f"  Cluster {label} labeled: '{consolidated_tag}' (based on {len(cluster_annotations)} items)")
                    except Exception as e:
                        print(f"  Failed to label cluster {label}: {e}")
                        cluster_mappings[label] = f"Cluster {label}"
                        
            # Apply consolidated tags in the DB
            for ann in valid_annotations:
                ann.consolidated_tag = cluster_mappings.get(ann.cluster_id, f"Cluster {ann.cluster_id}")
                upsert_annotation(session, ann)
                
        else:
            print("HDBSCAN found 0 clusters. All items classified as noise. Assigning 'Reaction Only'.")
            for ann in valid_annotations:
                ann.consolidated_tag = "Reaction Only"
                upsert_annotation(session, ann)
                
    print("Stage 2 clustering and labeling completed.")
