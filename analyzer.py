import os
import asyncio
import json
import re
import time
from dataclasses import dataclass
from collections import Counter
from typing import Callable, Literal, Optional, List
from pydantic import BaseModel, Field
from tqdm import tqdm
import numpy as np
from sklearn.cluster import HDBSCAN
from openrouter import OpenRouter
from sqlmodel import select
from dotenv import load_dotenv

# Load environment variables at module import time
load_dotenv()

from database import (
    Session, engine, upsert_post, upsert_comment, get_unmapped_raw_tags,
    Annotation, upsert_annotation, update_consolidated_tags, Post, Comment
)

# Stage 1 Schema
class Stage1Output(BaseModel):
    sentiment: float = Field(
        description="Sentiment polarity score: strongly negative (-1.0) to strongly positive (1.0). Choose from: -1.0, -0.5, 0.0, 0.5, 1.0"
    )
    summary: str = Field(
        description="A concise 1-2 sentence summary of the post or comment."
    )
    raw_tag: str = Field(
        description="A primary 2-5 word descriptor theme that captures the reason for the reaction when available. Use 'Reaction Only' if no specific analytical theme."
    )
    is_substantive: bool = Field(
        description="Whether the text gives a specific reason for the audience reaction, not merely emotion or enthusiasm."
    )
    reception_reason: Optional[str] = Field(
        default=None,
        description="A concise 2-6 word reason explaining why the audience reacts this way, or null for non-substantive reactions."
    )

# Stage 2 Schema (Cluster Labeling from LLM)
class ClusterLabelOutput(BaseModel):
    consolidated_tag: str = Field(
        description="A concise 2-5 word descriptor category for this cluster of reactions."
    )
    explanation: str = Field(
        description="One or two concise sentences explaining why this consolidated tag fits the cluster."
    )


STAGE_1_SYSTEM_PROMPT = (
    "You are an expert audience reception analyst specializing in how viewers discuss "
    "serialized television relationships on Reddit. Analyze the given text and extract "
    "sentiment, a concise summary, whether the text is analytically substantive, "
    "and the specific reception reason when one exists.\n\n"
    "SENTIMENT — You MUST choose exactly one of these five values:\n"
    "  -1.0 = Strongly Negative (harsh criticism, frustration, anger)\n"
    "  -0.5 = Mildly Negative (disappointment, mild criticism, concern)\n"
    "   0.0 = Neutral or Mixed (balanced take, factual observation, ambivalent)\n"
    "   0.5 = Mildly Positive (enjoyment, casual praise, mild enthusiasm)\n"
    "   1.0 = Strongly Positive (passionate praise, strong emotional approval)\n\n"
    "SUMMARY — Write a concise 1-2 sentence summary capturing the author's core point. "
    "If the text only expresses affect, say that it expresses a reaction without a specific reason.\n\n"
    "SUBSTANCE CHECK — Set is_substantive to true ONLY if the text states or strongly implies "
    "a specific why/how reason for liking, disliking, defending, criticizing, or interpreting the couple. "
    "If the text merely expresses intensity, affection, dislike, excitement, sadness, shipping, "
    "parasocial attachment, or a meme-like reaction without a concrete reason, set is_substantive=false.\n\n"
    "RECEPTION REASON / RAW TAG — If is_substantive=true, provide a neutral 2-6 word "
    "reception_reason and matching raw_tag that capture the specific reason behind the reaction. "
    "Prefer tags like "
    "'earned emotional payoff', 'forced conflict writing', 'supportive partner dynamic', "
    "'inconsistent character motivation', or 'chemistry through banter' over generic tags "
    "like 'character behavior', 'character analysis', or 'relationship opinion'. "
    "Do not invent reasons that are not stated or strongly implied.\n\n"
    "If is_substantive=false, set reception_reason=null and raw_tag='Reaction Only'. "
    "Do not turn mere intensity, attachment, or enthusiasm into a substantive tag unless "
    "the text gives a concrete cause for that reaction.\n\n"
    "Respond ONLY with a raw JSON object — no markdown, no explanation:\n"
    '{"sentiment": <float>, "summary": "<string>", "is_substantive": <bool>, '
    '"reception_reason": <string|null>, "raw_tag": "<string>"}'
)

SENTIMENT_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
OPENROUTER_RETRY_ATTEMPTS = 3
OPENROUTER_RETRY_BASE_DELAY = 2.0


@dataclass(frozen=True)
class Stage1Item:
    item_id: str
    item_type: Literal["post", "comment"]
    text: str


@dataclass(frozen=True)
class Stage1Result:
    item_id: str
    item_type: Literal["post", "comment"]
    annotation: Optional[Annotation] = None
    error: Optional[Exception] = None


@dataclass(frozen=True)
class Stage2EmbeddingItem:
    item_id: str
    text: str


@dataclass(frozen=True)
class Stage2EmbeddingResult:
    item_id: str
    embedding: Optional[List[float]] = None
    error: Optional[Exception] = None


@dataclass(frozen=True)
class Stage2LabelItem:
    cluster_id: int
    prompt: str
    fallback_label: str
    fallback_explanation: str
    item_count: int


@dataclass(frozen=True)
class Stage2LabelResult:
    cluster_id: int
    label: str
    explanation: str
    item_count: int
    error: Optional[Exception] = None


def parse_json_content(content: str) -> dict:
    """Helper to clean up markdown codeblocks if any, and parse content as JSON."""
    clean_content = content.strip()
    if clean_content.startswith("```"):
        lines = clean_content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        clean_content = "\n".join(lines).strip()
    return json.loads(clean_content)


def normalize_stage_1_output(parsed_data: dict) -> Stage1Output:
    try:
        raw_sentiment = float(parsed_data.get("sentiment", 0.0))
    except (ValueError, TypeError):
        raw_sentiment = 0.0
    parsed_data["sentiment"] = min(SENTIMENT_VALUES, key=lambda x: abs(x - raw_sentiment))
    if "is_substantive" not in parsed_data:
        parsed_data["is_substantive"] = str(parsed_data.get("raw_tag", "")).strip().lower() != "reaction only"
    if "reception_reason" not in parsed_data:
        parsed_data["reception_reason"] = parsed_data.get("raw_tag") if parsed_data["is_substantive"] else None
    return Stage1Output.model_validate(parsed_data)


def normalize_substance_fields(output: Stage1Output) -> Stage1Output:
    reason = clean_label_text(output.reception_reason)
    raw_tag = clean_label_text(output.raw_tag)
    is_substantive = bool(output.is_substantive)

    if not is_substantive or not reason or raw_tag.lower() == "reaction only":
        output.is_substantive = False
        output.reception_reason = None
        output.raw_tag = "Reaction Only"
        return output

    output.reception_reason = reason
    output.raw_tag = raw_tag or reason
    return output


def clean_label_text(label: Optional[str]) -> str:
    if not label:
        return ""
    label = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(label))
    label = re.sub(r"\s+", " ", label).strip(" \"'`.,;:-")
    return label


def clean_explanation_text(explanation: Optional[str]) -> str:
    if not explanation:
        return ""
    explanation = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(explanation))
    explanation = re.sub(r"\s+", " ", explanation).strip(" \"'`")
    return explanation


def is_generic_cluster_label(label: str) -> bool:
    cleaned = clean_label_text(label).lower()
    if not cleaned:
        return True
    return bool(re.fullmatch(r"cluster\s*[a-z0-9_-]*", cleaned))


def fallback_cluster_label(annotations: List[Annotation]) -> str:
    tags = [
        clean_label_text(ann.reception_reason or ann.raw_tag)
        for ann in annotations
        if clean_label_text(ann.reception_reason or ann.raw_tag)
    ]
    tags = [tag for tag in tags if tag.lower() != "reaction only"]
    if not tags:
        return "Reaction Only"
    return Counter(tags).most_common(1)[0][0]


def fallback_cluster_explanation(label: str, annotations: List[Annotation]) -> str:
    summaries = [clean_label_text(ann.summary) for ann in annotations if clean_label_text(ann.summary)]
    if summaries:
        return f"This cluster is labeled '{label}' because representative items share this pattern: {summaries[0]}"
    return f"This cluster is labeled '{label}' based on the representative raw tags in the cluster."


def is_retryable_openrouter_error(error: Exception) -> bool:
    message = str(error).lower()
    retry_markers = [
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "timeout",
        "temporarily",
        "connection",
        "server error",
        "502",
        "503",
        "504",
    ]
    return any(marker in message for marker in retry_markers)


def run_openrouter_with_retries(operation: Callable[[], object], context: str):
    last_error = None
    for attempt in range(1, OPENROUTER_RETRY_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as e:
            last_error = e
            if attempt >= OPENROUTER_RETRY_ATTEMPTS or not is_retryable_openrouter_error(e):
                raise
            delay = OPENROUTER_RETRY_BASE_DELAY * attempt
            tqdm.write(f"{context}: retrying after transient OpenRouter error ({attempt}/{OPENROUTER_RETRY_ATTEMPTS}): {e}")
            time.sleep(delay)
    raise last_error


def analyze_stage_1_item(
    item: Stage1Item,
    openrouter_api_key: str,
    model_name: str,
    temperature: float,
) -> Stage1Result:
    try:
        def request():
            with OpenRouter(api_key=openrouter_api_key) as client:
                return client.chat.send(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": STAGE_1_SYSTEM_PROMPT},
                        {"role": "user", "content": item.text}
                    ],
                    temperature=temperature
                )

        response = run_openrouter_with_retries(request, f"Stage 1 {item.item_type} {item.item_id}")
        content = response.choices[0].message.content
        output = normalize_substance_fields(normalize_stage_1_output(parse_json_content(content)))
        annotation = Annotation(
            item_id=item.item_id,
            item_type=item.item_type,
            sentiment=output.sentiment,
            summary=output.summary,
            raw_tag=output.raw_tag,
            is_substantive=output.is_substantive,
            reception_reason=output.reception_reason,
        )
        return Stage1Result(item_id=item.item_id, item_type=item.item_type, annotation=annotation)
    except Exception as e:
        return Stage1Result(item_id=item.item_id, item_type=item.item_type, error=e)


async def run_stage_1_item_batch(
    items: List[Stage1Item],
    openrouter_api_key: str,
    model_name: str,
    temperature: float,
    concurrency: int,
    desc: str,
) -> List[Stage1Result]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(item: Stage1Item) -> Stage1Result:
        async with semaphore:
            return await asyncio.to_thread(
                analyze_stage_1_item,
                item,
                openrouter_api_key,
                model_name,
                temperature,
            )

    tasks = [asyncio.create_task(run_one(item)) for item in items]
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
        results.append(await task)
    return results


def generate_stage_2_embedding(
    item: Stage2EmbeddingItem,
    openrouter_api_key: str,
    embedding_model: str,
) -> Stage2EmbeddingResult:
    try:
        def request():
            with OpenRouter(api_key=openrouter_api_key) as client:
                return client.embeddings.generate(
                    model=embedding_model,
                    input=item.text
                )

        res = run_openrouter_with_retries(request, f"Stage 2 embedding {item.item_id}")
        return Stage2EmbeddingResult(item_id=item.item_id, embedding=res.data[0].embedding)
    except Exception as e:
        return Stage2EmbeddingResult(item_id=item.item_id, error=e)


async def run_stage_2_embedding_batch(
    items: List[Stage2EmbeddingItem],
    openrouter_api_key: str,
    embedding_model: str,
    concurrency: int,
) -> List[Stage2EmbeddingResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(item: Stage2EmbeddingItem) -> Stage2EmbeddingResult:
        async with semaphore:
            return await asyncio.to_thread(
                generate_stage_2_embedding,
                item,
                openrouter_api_key,
                embedding_model,
            )

    tasks = [asyncio.create_task(run_one(item)) for item in items]
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating Embeddings"):
        results.append(await task)
    return results


def label_stage_2_cluster(
    item: Stage2LabelItem,
    openrouter_api_key: str,
    labeling_model: str,
) -> Stage2LabelResult:
    try:
        def request():
            with OpenRouter(api_key=openrouter_api_key) as client:
                return client.chat.send(
                    model=labeling_model,
                    messages=[
                        {"role": "user", "content": item.prompt}
                    ],
                    temperature=0.1
                )

        response = run_openrouter_with_retries(request, f"Stage 2 cluster {item.cluster_id}")
        content = response.choices[0].message.content
        parsed_data = parse_json_content(content)
        output = ClusterLabelOutput.model_validate(parsed_data)
        label = clean_label_text(output.consolidated_tag)
        if is_generic_cluster_label(label):
            label = item.fallback_label
        explanation = clean_explanation_text(output.explanation) or item.fallback_explanation
        return Stage2LabelResult(
            cluster_id=item.cluster_id,
            label=label,
            explanation=explanation,
            item_count=item.item_count,
        )
    except Exception as e:
        return Stage2LabelResult(
            cluster_id=item.cluster_id,
            label=item.fallback_label,
            explanation=item.fallback_explanation,
            item_count=item.item_count,
            error=e,
        )


async def run_stage_2_label_batch(
    items: List[Stage2LabelItem],
    openrouter_api_key: str,
    labeling_model: str,
    concurrency: int,
) -> List[Stage2LabelResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(item: Stage2LabelItem) -> Stage2LabelResult:
        async with semaphore:
            return await asyncio.to_thread(
                label_stage_2_cluster,
                item,
                openrouter_api_key,
                labeling_model,
            )

    tasks = [asyncio.create_task(run_one(item)) for item in items]
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Labeling Clusters"):
        results.append(await task)
    return results


def run_stage_1_analysis(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    force_reanalyze: bool = False,
    concurrency: int = 10
):
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for Stage 1 analysis.")
    
    if not model_name:
        model_name = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it")
    
    with Session(engine) as session:
        if force_reanalyze:
            posts_stmt = select(Post).where((Post.status == "pending") | (Post.status == "processed"))
            comments_stmt = select(Comment).where((Comment.status == "pending") | (Comment.status == "processed"))
        else:
            posts_stmt = select(Post).where(Post.status == "pending")
            comments_stmt = select(Comment).where(Comment.status == "pending")

        posts_to_process = session.exec(posts_stmt).all()
        comments_to_process = session.exec(comments_stmt).all()

    post_items = [
        Stage1Item(
            item_id=post.id,
            item_type="post",
            text=f"Title: {post.title}\n\nBody: {post.selftext}",
        )
        for post in posts_to_process
    ]
    comment_items = [
        Stage1Item(item_id=comment.id, item_type="comment", text=comment.body)
        for comment in comments_to_process
    ]

    print(f"Stage 1: Processing {len(post_items)} posts with concurrency {max(1, concurrency)}...")
    post_results = asyncio.run(
        run_stage_1_item_batch(
            post_items,
            openrouter_api_key,
            model_name,
            temperature,
            concurrency,
            "Analyzing Posts",
        )
    )

    print(f"Stage 1: Processing {len(comment_items)} comments with concurrency {max(1, concurrency)}...")
    comment_results = asyncio.run(
        run_stage_1_item_batch(
            comment_items,
            openrouter_api_key,
            model_name,
            temperature,
            concurrency,
            "Analyzing Comments",
        )
    )

    with Session(engine) as session:
        for result in post_results + comment_results:
            if result.annotation:
                upsert_annotation(session, result.annotation)

            if result.item_type == "post":
                item = session.get(Post, result.item_id)
                upsert_item = upsert_post
            else:
                item = session.get(Comment, result.item_id)
                upsert_item = upsert_comment

            if not item:
                continue
            if result.error:
                print(f"Failed to process {result.item_type} {result.item_id}: {result.error}")
                item.status = "failed"
            else:
                item.status = "processed"
            upsert_item(session, item)

    print("Stage 1 feature extraction completed.")

def run_stage_2_analysis(
    embedding_model: str = "sentence-transformers/all-minilm-l12-v2",
    labeling_model: str = "google/gemma-4-26b-a4b-it",
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    force_reembed: bool = False,
    force_recluster: bool = False,
    label_sample_size: int = 15,
    concurrency: int = 10
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
        substantive_annotations = [ann for ann in annotations if ann.is_substantive]
        reaction_only_count = len(annotations) - len(substantive_annotations)
        if reaction_only_count:
            print(f"Stage 2: Excluding {reaction_only_count} non-substantive Reaction Only annotations from clustering.")

        for ann in annotations:
            if not ann.is_substantive:
                ann.cluster_id = None
                ann.consolidated_tag = None
                ann.cluster_explanation = None
                upsert_annotation(session, ann)

        if not substantive_annotations:
            print("No substantive annotations found. Stage 2 clustering skipped.")
            return
        
        # 1. Generate Embeddings (checking cache)
        embeddings_needed = [
            Stage2EmbeddingItem(
                item_id=ann.item_id,
                text=f"Reception Reason: {ann.reception_reason or ann.raw_tag} | Raw Tag: {ann.raw_tag} | Summary: {ann.summary}",
            )
            for ann in substantive_annotations
            if force_reembed or not ann.embedding
        ]

        if embeddings_needed:
            print(
                f"Generating embeddings for {len(embeddings_needed)} annotations "
                f"using {embedding_model} with concurrency {max(1, concurrency)}..."
            )
            embedding_results = asyncio.run(
                run_stage_2_embedding_batch(
                    embeddings_needed,
                    openrouter_api_key,
                    embedding_model,
                    concurrency,
                )
            )
            for result in embedding_results:
                ann = session.get(Annotation, result.item_id)
                if not ann:
                    continue
                if result.embedding:
                    ann.embedding = json.dumps(result.embedding)
                    upsert_annotation(session, ann)
                elif result.error:
                    print(f"Failed to generate embedding for {result.item_id}: {result.error}")

            # Reload annotations to ensure we have all embeddings
            annotations = session.exec(annotations_stmt).all()
            substantive_annotations = [ann for ann in annotations if ann.is_substantive]
        
        # Filter annotations that have embeddings
        valid_annotations = [ann for ann in substantive_annotations if ann.embedding is not None]
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
            # Assign a local fallback label without calling the cluster labeler.
            for ann in valid_annotations:
                fallback_label = fallback_cluster_label([ann])
                ann.cluster_id = 0
                ann.consolidated_tag = fallback_label
                ann.cluster_explanation = fallback_cluster_explanation(fallback_label, [ann])
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
            label_items = []
            for label in valid_labels:
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
                    sample_text += (
                        f"- Reception Reason: '{item.reception_reason or item.raw_tag}' | "
                        f"Raw Tag: '{item.raw_tag}' | Summary: '{item.summary}'\n"
                    )

                prompt = (
                    "You are an expert audience reception analyst studying how viewers discuss "
                    "serialized television relationships.\n"
                    "Below are representative reception reasons, raw tags, and summaries from a semantically clustered "
                    f"group of {len(cluster_annotations)} audience reactions:\n\n"
                    f"{sample_text}\n"
                    "Identify the unifying theme across these reactions and produce a single "
                    "consolidated tag (2-5 words) that captures the shared reception reason, "
                    "not merely the fact that commenters are emotionally engaged. "
                    "Prefer labels like 'earned emotional payoff', 'forced conflict writing', "
                    "'natural romantic chemistry', or 'inconsistent character motivation' over "
                    "generic labels like 'Character Analysis' or 'Relationship Dynamics'. "
                    "Stay neutral and do not infer a reason that is not present in the examples. "
                    "Never respond with a placeholder such as 'Cluster 1'.\n\n"
                    "Respond ONLY with a raw JSON object — no markdown, no explanation:\n"
                    '{"consolidated_tag": "<your label>", "explanation": "<one or two sentence explanation>"}'
                )
                fallback_label = fallback_cluster_label(cluster_annotations)
                label_items.append(
                    Stage2LabelItem(
                        cluster_id=int(label),
                        prompt=prompt,
                        fallback_label=fallback_label,
                        fallback_explanation=fallback_cluster_explanation(fallback_label, cluster_annotations),
                        item_count=len(cluster_annotations),
                    )
                )

            label_results = asyncio.run(
                run_stage_2_label_batch(
                    label_items,
                    openrouter_api_key,
                    labeling_model,
                    concurrency,
                )
            )
            cluster_mappings = {}
            cluster_explanations = {}
            for result in label_results:
                cluster_mappings[result.cluster_id] = result.label
                cluster_explanations[result.cluster_id] = result.explanation
                if result.error:
                    print(f"  Failed to label cluster {result.cluster_id}: {result.error}")
                print(f"  Cluster {result.cluster_id} labeled: '{result.label}' (based on {result.item_count} items)")
                        
            # Apply consolidated tags in the DB
            for ann in valid_annotations:
                ann.consolidated_tag = cluster_mappings.get(ann.cluster_id, fallback_cluster_label([ann]))
                ann.cluster_explanation = cluster_explanations.get(
                    ann.cluster_id,
                    fallback_cluster_explanation(ann.consolidated_tag, [ann]),
                )
                upsert_annotation(session, ann)
                
        else:
            print("HDBSCAN found 0 clusters. All substantive items classified as noise. Assigning fallback labels.")
            for ann in valid_annotations:
                fallback_label = fallback_cluster_label([ann])
                ann.consolidated_tag = fallback_label
                ann.cluster_explanation = (
                    "HDBSCAN found no stable clusters, so this item keeps its strongest available "
                    f"reception reason: {fallback_label}."
                )
                upsert_annotation(session, ann)
                
    print("Stage 2 clustering and labeling completed.")
