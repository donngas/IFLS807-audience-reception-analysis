import os
import json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sqlmodel import Session, select

# Prevent matplotlib from showing GUI window immediately or failing in headless setups
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from database import Annotation, Post, Comment, engine

# Set default seaborn style for beautiful aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def load_annotation_dataframe(session: Session) -> pd.DataFrame:
    """Load annotations from database into a pandas DataFrame."""
    statement = select(Annotation)
    annotations = session.exec(statement).all()
    if not annotations:
        return pd.DataFrame()
    
    data = []
    for ann in annotations:
        data.append({
            "item_id": ann.item_id,
            "item_type": ann.item_type,
            "sentiment": ann.sentiment,
            "summary": ann.summary,
            "raw_tag": ann.raw_tag,
            "consolidated_tag": ann.consolidated_tag or "Unclustered",
            "cluster_id": ann.cluster_id if ann.cluster_id is not None else -1,
            "embedding": ann.embedding
        })
    return pd.DataFrame(data)

def load_temporal_dataframe(session: Session) -> pd.DataFrame:
    """Load annotations combined with created_utc and subreddit metadata."""
    # 1. Fetch Post Annotations
    post_stmt = select(
        Annotation.item_id,
        Annotation.sentiment,
        Annotation.consolidated_tag,
        Post.created_utc,
        Post.subreddit
    ).join(Post, Annotation.item_id == Post.id).where(Annotation.item_type == "post")
    posts_data = session.exec(post_stmt).all()
    
    # 2. Fetch Comment Annotations (joining with Comment and its Post to get the subreddit)
    comment_stmt = select(
        Annotation.item_id,
        Annotation.sentiment,
        Annotation.consolidated_tag,
        Comment.created_utc,
        Post.subreddit
    ).join(Comment, Annotation.item_id == Comment.id).join(Post, Comment.post_id == Post.id).where(Annotation.item_type == "comment")
    comments_data = session.exec(comment_stmt).all()
    
    combined = []
    for item_id, sentiment, tag, utc, sub in posts_data + comments_data:
        combined.append({
            "item_id": item_id,
            "sentiment": sentiment,
            "consolidated_tag": tag or "Unclustered",
            "created_utc": utc,
            "date": datetime.fromtimestamp(utc),
            "subreddit": sub
        })
        
    return pd.DataFrame(combined)

def handle_output(fig, save_path: Optional[str] = None):
    """Show the figure or save it to file based on user choice."""
    if save_path:
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        # Save figure
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Visualization exported successfully to: {save_path}")
    else:
        # Preview
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying GUI plot window: {e}")
            print("Tip: If you are running in a headless environment, choose to export to a file instead.")

def plot_semantic_map(session: Session, save_path: Optional[str] = None):
    """Generate a 2D t-SNE plot of annotations colored by consolidated tag."""
    df = load_annotation_dataframe(session)
    if df.empty:
        print("No data available to plot.")
        return
        
    # Filter for items with embeddings
    df_valid = df[df["embedding"].notna()].copy()
    if len(df_valid) < 2:
        print("Need at least 2 annotated items with embeddings to generate a semantic map.")
        return
        
    # Reconstruct embedding matrix
    embeddings = np.array([json.loads(emb) for emb in df_valid["embedding"]])
    
    # Run t-SNE (force perplexity adjustment if sample size is very small)
    perplexity = min(30, max(1, len(df_valid) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
    X_2d = tsne.fit_transform(embeddings)
    
    df_valid["x"] = X_2d[:, 0]
    df_valid["y"] = X_2d[:, 1]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.scatterplot(
        data=df_valid,
        x="x",
        y="y",
        hue="consolidated_tag",
        palette="viridis",
        alpha=0.8,
        s=60,
        ax=ax
    )
    
    ax.set_title("Semantic Mapping of Audience Reception Themes (t-SNE)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Themes")
    
    handle_output(fig, save_path)

def plot_sentiment_by_theme(session: Session, save_path: Optional[str] = None):
    """Generate a box/violin plot of sentiment distribution per consolidated tag."""
    df = load_annotation_dataframe(session)
    if df.empty:
        print("No data available to plot.")
        return
        
    # Filter for processed items
    df_valid = df[df["consolidated_tag"].notna()].copy()
    if df_valid.empty:
        print("No annotated items found to show theme sentiment distributions.")
        return
        
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by tag and count occurrences for labelling
    counts = df_valid["consolidated_tag"].value_counts()
    order = counts.index
    
    sns.boxplot(
        data=df_valid,
        y="consolidated_tag",
        x="sentiment",
        order=order,
        palette="coolwarm",
        ax=ax,
        hue="consolidated_tag",
        legend=False
    )
    
    # Add count labels next to the y-tick labels
    yticks_labels = [f"{label} (n={counts[label]})" for label in order]
    ax.set_yticklabels(yticks_labels)
    
    ax.set_title("Sentiment Distribution by Consolidated Theme", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sentiment Score (Negative -1.0 to Positive 1.0)", fontsize=11)
    ax.set_ylabel("Consolidated Theme", fontsize=11)
    ax.set_xlim(-1.1, 1.1)
    
    handle_output(fig, save_path)

def plot_theme_trends_over_time(session: Session, save_path: Optional[str] = None, bin_by: str = "W"):
    """Plot trends of top consolidated tags over time."""
    df = load_temporal_dataframe(session)
    if df.empty:
        print("No temporal data available to plot.")
        return
        
    # Sort and bin by time period (defaults to weekly)
    df = df.sort_values("date")
    
    # Group by date bins and tag counts
    df["period"] = df["date"].dt.to_period(bin_by).dt.to_timestamp()
    
    # Pivot table to count occurrences
    pivot = df.pivot_table(index="period", columns="consolidated_tag", values="item_id", aggfunc="count", fill_value=0)
    
    # Filter to show only the top 8 tags to avoid visual clutter
    top_tags = df["consolidated_tag"].value_counts().head(8).index
    pivot_filtered = pivot[pivot.columns.intersection(top_tags)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in pivot_filtered.columns:
        ax.plot(pivot_filtered.index, pivot_filtered[col], label=col, marker="o", linewidth=2)
        
    ax.set_title(f"Thematic Trends Over Time (Binned: {bin_by})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Time Period", fontsize=11)
    ax.set_ylabel("Volume (Count of Posts/Comments)", fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Themes")
    
    # Beautiful date formatting
    fig.autofmt_xdate()
    
    handle_output(fig, save_path)

def plot_subreddit_theme_distribution(session: Session, save_path: Optional[str] = None):
    """Plot theme composition across different subreddits."""
    df = load_temporal_dataframe(session)
    if df.empty:
        print("No subreddit data available to plot.")
        return
        
    # Group by subreddit and tag
    ct = pd.crosstab(df["subreddit"], df["consolidated_tag"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ct.plot(kind="bar", stacked=True, colormap="viridis", ax=ax)
    
    ax.set_title("Thematic Theme Composition by Subreddit", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Subreddit", fontsize=11)
    ax.set_ylabel("Volume (Count)", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Themes")
    
    handle_output(fig, save_path)
