import os
import json
import re
import textwrap
from typing import Optional

import numpy as np
import pandas as pd
from sqlmodel import Session, select

# Prevent matplotlib from showing GUI window immediately or failing in headless setups
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from database import Annotation

SENTIMENT_BUCKETS = [-1.0, -0.5, 0.0, 0.5, 1.0]
SENTIMENT_LABELS = {
    -1.0: "Strongly\nNegative",
    -0.5: "Mildly\nNegative",
    0.0: "Neutral/\nMixed",
    0.5: "Mildly\nPositive",
    1.0: "Strongly\nPositive",
}
SENTIMENT_COLORS = {
    -1.0: "#b2182b",
    -0.5: "#ef8a62",
    0.0: "#d9d9d9",
    0.5: "#67a9cf",
    1.0: "#2166ac",
}

# Set default seaborn style for beautiful aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.autolayout"] = False


def clean_plot_label(label: Optional[str]) -> str:
    if not label:
        return "Unclustered"
    label = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(label))
    label = re.sub(r"\s+", " ", label).strip()
    return label or "Unclustered"


def wrap_label(label: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(clean_plot_label(label), width=width)) or "Unclustered"


def add_inside_legend(ax, title: str = "Themes", max_columns: int = 1):
    legend = ax.legend(
        title=title,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        framealpha=0.92,
        fontsize=8,
        title_fontsize=9,
        borderaxespad=0.4,
        ncol=max_columns,
    )
    if legend:
        for text in legend.get_texts():
            text.set_text(wrap_label(text.get_text(), width=24))
    return legend


def finish_figure(fig, title: str):
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    fig.tight_layout()


def compact_theme_values(df: pd.DataFrame, column: str = "consolidated_tag", top_n: int = 10) -> pd.Series:
    counts = df[column].value_counts()
    top_values = set(counts.head(top_n).index)
    return df[column].where(df[column].isin(top_values), "Other themes")

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
            "consolidated_tag": clean_plot_label(ann.consolidated_tag),
            "cluster_id": ann.cluster_id if ann.cluster_id is not None else -1,
            "embedding": ann.embedding
        })
    return pd.DataFrame(data)

def handle_output(fig, save_path: Optional[str] = None):
    """Show the figure or save it to file based on user choice."""
    if save_path:
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(save_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        # Save figure
        finish_figure(fig, os.path.basename(save_path))
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Visualization exported successfully to: {save_path}")
    else:
        # Preview
        try:
            finish_figure(fig, fig._suptitle.get_text() if fig._suptitle else "Visualization")
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
    
    df_valid["theme_display"] = compact_theme_values(df_valid, top_n=10)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7.5), constrained_layout=True)
    sns.scatterplot(
        data=df_valid,
        x="x",
        y="y",
        hue="theme_display",
        palette="viridis",
        alpha=0.8,
        s=60,
        ax=ax
    )
    
    ax.set_title("Semantic Mapping of Audience Reception Themes (t-SNE)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    add_inside_legend(ax)
    
    handle_output(fig, save_path)

def plot_sentiment_by_theme(session: Session, save_path: Optional[str] = None):
    """Generate a normalized stacked bar chart of sentiment distribution per theme."""
    df = load_annotation_dataframe(session)
    if df.empty:
        print("No data available to plot.")
        return
        
    # Filter for processed items
    df_valid = df[df["consolidated_tag"].notna()].copy()
    if df_valid.empty:
        print("No annotated items found to show theme sentiment distributions.")
        return

    df_valid["sentiment_bucket"] = df_valid["sentiment"].apply(
        lambda value: min(SENTIMENT_BUCKETS, key=lambda bucket: abs(bucket - float(value)))
    )
    counts = df_valid["consolidated_tag"].value_counts()
    top_themes = counts.head(20).index
    df_valid = df_valid[df_valid["consolidated_tag"].isin(top_themes)].copy()

    count_table = pd.crosstab(df_valid["consolidated_tag"], df_valid["sentiment_bucket"])
    count_table = count_table.reindex(index=top_themes, columns=SENTIMENT_BUCKETS, fill_value=0)
    percent_table = count_table.div(count_table.sum(axis=1), axis=0).fillna(0) * 100
    theme_labels = [wrap_label(f"{theme} (n={int(counts[theme])})", width=34) for theme in percent_table.index]

    fig_height = min(14, max(6, 0.45 * len(percent_table) + 2))
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
    left = np.zeros(len(percent_table))
    for bucket in SENTIMENT_BUCKETS:
        values = percent_table[bucket].values
        ax.barh(
            theme_labels,
            values,
            left=left,
            color=SENTIMENT_COLORS[bucket],
            label=SENTIMENT_LABELS[bucket].replace("\n", " "),
        )
        left += values

    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title("Sentiment Composition by Consolidated Theme", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Share Within Theme (%)", fontsize=11)
    ax.set_ylabel("Consolidated Theme", fontsize=11)
    ax.set_xlim(0, 100)
    ax.legend(title="Sentiment", loc="lower right", frameon=True, framealpha=0.92, fontsize=8, title_fontsize=9)
    
    handle_output(fig, save_path)


def plot_sentiment_distribution(session: Session, save_path: Optional[str] = None):
    """Plot the discrete sentiment score distribution overall and by item type."""
    df = load_annotation_dataframe(session)
    if df.empty:
        print("No sentiment annotations available to plot.")
        return

    df = df.copy()
    df["sentiment_bucket"] = df["sentiment"].apply(
        lambda value: min(SENTIMENT_BUCKETS, key=lambda bucket: abs(bucket - float(value)))
    )
    rows = []
    for item_type, frame in [("All", df), ("Posts", df[df["item_type"] == "post"]), ("Comments", df[df["item_type"] == "comment"])]:
        counts = frame["sentiment_bucket"].value_counts()
        total = counts.sum()
        for bucket in SENTIMENT_BUCKETS:
            count = int(counts.get(bucket, 0))
            rows.append({
                "item_type": item_type,
                "sentiment": bucket,
                "sentiment_label": SENTIMENT_LABELS[bucket],
                "count": count,
                "percent": (count / total * 100) if total else 0.0,
            })
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    sns.barplot(
        data=plot_df,
        x="sentiment_label",
        y="percent",
        hue="item_type",
        palette=["#4c78a8", "#72b7b2", "#f58518"],
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{bar.get_height():.1f}%" for bar in container], padding=3, fontsize=8)

    ax.set_title("Relative Sentiment Score Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sentiment Score", fontsize=11)
    ax.set_ylabel("Share Within Item Type (%)", fontsize=11)
    ax.set_ylim(0, max(100, plot_df["percent"].max() * 1.18))
    ax.legend(title="Item Type", loc="upper right", frameon=True, framealpha=0.92)
    ax.margins(y=0.15)

    handle_output(fig, save_path)

def plot_theme_dominance_bar(session: Session, save_path: Optional[str] = None, top_n: int = 20):
    """Plot raw counts and percentages for the most common consolidated themes."""
    df = load_annotation_dataframe(session)
    if df.empty:
        print("No data available to plot.")
        return

    counts = df["consolidated_tag"].value_counts()
    if counts.empty:
        print("No clustered themes available to plot.")
        return

    top_counts = counts.head(top_n)
    total = counts.sum()
    plot_df = pd.DataFrame({
        "theme": [wrap_label(label, width=34) for label in top_counts.index],
        "count": top_counts.values,
        "percent": top_counts.values / total * 100,
    })

    fig_height = min(14, max(6, 0.42 * len(plot_df) + 2))
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)
    colors = sns.color_palette("viridis", len(plot_df))
    bars = ax.barh(plot_df["theme"], plot_df["count"], color=colors)
    ax.invert_yaxis()

    labels = [f"{row.percent:.1f}%" for row in plot_df.itertuples()]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=9)

    ax.set_title("Theme Dominance by Count", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Items in Theme", fontsize=11)
    ax.set_ylabel("Consolidated Theme", fontsize=11)
    ax.margins(x=0.12)
    ax.tick_params(axis="y", labelsize=9)

    handle_output(fig, save_path)


def plot_theme_dominance_pareto(session: Session, save_path: Optional[str] = None, top_n: int = 20):
    """Plot theme counts with cumulative share to show concentration/dominance."""
    df = load_annotation_dataframe(session)
    if df.empty:
        print("No data available to plot.")
        return

    counts = df["consolidated_tag"].value_counts().head(top_n)
    if counts.empty:
        print("No clustered themes available to plot.")
        return

    total = df["consolidated_tag"].value_counts().sum()
    x_labels = [wrap_label(label, width=18) for label in counts.index]
    cumulative = counts.cumsum() / total * 100

    fig, ax1 = plt.subplots(figsize=(13, 7), constrained_layout=True)
    bars = ax1.bar(range(len(counts)), counts.values, color=sns.color_palette("viridis", len(counts)))
    ax1.set_ylabel("Items in Theme", fontsize=11)
    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_title("Theme Dominance Pareto View", fontsize=14, fontweight="bold", pad=15)

    ax2 = ax1.twinx()
    ax2.plot(range(len(counts)), cumulative.values, color="#c43c39", marker="o", linewidth=2)
    ax2.set_ylabel("Cumulative Share (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(False)

    ax1.bar_label(bars, padding=3, fontsize=8)
    
    handle_output(fig, save_path)
