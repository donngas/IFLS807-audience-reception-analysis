import os
import sys
import argparse
from dotenv import load_dotenv
import questionary
from sqlmodel import Session, select

from database import (
    init_db, engine, Post, Comment, Annotation, get_statistics,
    reset_failed_items, reset_processed_items, clear_stage_2_clustering,
    reset_specific_items, nuke_database
)
from scraper import scrape_reddit
from analyzer import run_stage_1_analysis, run_stage_2_analysis
from util import export_to_csv, export_to_json, print_stats
from visualization import (
    plot_semantic_map, plot_sentiment_by_theme,
    plot_theme_trends_over_time, plot_subreddit_theme_distribution
)

def main():
    load_dotenv()
    init_db()
    
    parser = argparse.ArgumentParser(description="Audience Reception Analysis Pipeline")
    parser.add_argument("--action", choices=["scrape", "analyze", "cluster", "export", "stats"], help="Action to perform")
    
    # Scrape args
    parser.add_argument("--query", type=str, help="Search query for Reddit")
    parser.add_argument("--subreddits", type=str, help="Comma-separated list of subreddits")
    parser.add_argument("--post-limit", type=int, default=100, help="Max posts to scrape")
    parser.add_argument("--comment-limit", type=int, default=100, help="Max comments per post to scrape")
    parser.add_argument("--sort", type=str, default="top", help="Reddit search sort")
    parser.add_argument("--time-filter", type=str, default="all", help="Reddit search time filter")
    parser.add_argument("--force-overwrite", action="store_true", help="Force overwrite existing scraped items")
    
    # Stage 1 args
    parser.add_argument("--model", type=str, help="OpenRouter model name")
    parser.add_argument("--batch-limit", type=int, default=100, help="Max items to process in Stage 1")
    parser.add_argument("--temp", type=float, default=0.1, help="Model temperature")
    parser.add_argument("--force-reanalyze", action="store_true", help="Force re-analyze processed items")
    
    # Stage 2 args
    parser.add_argument("--embed-model", type=str, default="sentence-transformers/all-minilm-l12-v2", help="Embedding model")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size")
    parser.add_argument("--force-reembed", action="store_true", help="Force regenerate embeddings")
    parser.add_argument("--force-recluster", action="store_true", help="Force re-run clustering and labeling")
    
    # Export args
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Export format")
    parser.add_argument("--output", type=str, help="Output file path for export")
    
    args = parser.parse_args()
    
    # If no arguments provided, use interactive wizard
    if len(sys.argv) == 1:
        run_interactive_wizard()
        return

    # CLI Execution
    if args.action == "scrape":
        if not args.query:
            print("Error: --query is required for scrape action.")
            return
        sub_list = [s.strip() for s in args.subreddits.split(",")] if args.subreddits else None
        scrape_reddit(args.query, sub_list, args.post_limit, args.comment_limit, sort=args.sort, time_filter=args.time_filter, skip_existing=not args.force_overwrite)
        
    elif args.action == "analyze":
        print("Starting Stage 1 Analysis (Feature Extraction)...")
        run_stage_1_analysis(model_name=args.model, limit=args.batch_limit, temperature=args.temp, force_reanalyze=args.force_reanalyze)
        
    elif args.action == "cluster":
        print("Starting Stage 2 Analysis (Semantic Embedding Clustering)...")
        run_stage_2_analysis(embedding_model=args.embed_model, labeling_model=args.model or "google/gemma-4-26b-a4b-it", min_cluster_size=args.min_cluster_size, force_reembed=args.force_reembed, force_recluster=args.force_recluster)
        
    elif args.action == "export":
        if not args.output:
            print("Error: --output is required for export action.")
            return
        if args.format == "csv":
            export_to_csv(args.output)
        else:
            export_to_json(args.output)
            
    elif args.action == "stats":
        print_stats()

def run_interactive_wizard():
    print("Welcome to the Audience Reception Analysis Pipeline!\n")
    
    while True:
        choice = questionary.select(
            "What would you like to do?",
            choices=[
                "1. Scrape Reddit posts & comments",
                "2. Run Stage 1 Analysis (Sentiment & Tags)",
                "3. Run Stage 2 Analysis (Embedding & HDBSCAN Clustering)",
                "4. View Database Records & Stats",
                "5. Manage / Reset Database Records",
                "6. Export data to CSV/JSON",
                "7. Run Visualizations & Dashboards",
                "8. Exit"
            ]
        ).ask()
        
        if not choice or choice.startswith("8"):
            print("Exiting pipeline. Goodbye!")
            break
            
        if choice.startswith("1"):
            query = questionary.text("Enter search query (e.g. {Jake AND Amy}):").ask()
            if not query:
                continue
                
            sub_str = questionary.text("Enter subreddits (comma-separated, leave blank for all):").ask()
            sub_list = [s.strip() for s in sub_str.split(",")] if sub_str else None
            
            p_limit = questionary.text("Post limit:", default="100").ask()
            c_limit = questionary.text("Comment limit per post:", default="100").ask()
            
            # Advanced Scraping Options
            sort_val = "top"
            time_val = "all"
            skip_val = True
            adv = questionary.confirm("Configure advanced scraping options?", default=False).ask()
            if adv:
                sort_val = questionary.select("Sort by:", choices=["top", "hot", "new", "relevance"], default="top").ask()
                time_val = questionary.select("Time filter:", choices=["all", "day", "week", "month", "year"], default="all").ask()
                skip_val = questionary.confirm("Skip already scraped posts?", default=True).ask()
                
            scrape_reddit(query, sub_list, int(p_limit), int(c_limit), sort=sort_val, time_filter=time_val, skip_existing=skip_val)
            
        elif choice.startswith("2"):
            # Default or Advanced Stage 1
            model_val = None
            limit_val = 100
            temp_val = 0.1
            force_val = False
            adv = questionary.confirm("Configure advanced Stage 1 options?", default=False).ask()
            if adv:
                model_val = questionary.text("OpenRouter Model:", default="google/gemma-4-26b-a4b-it").ask()
                limit_val = int(questionary.text("Batch limit (max items to process):", default="100").ask())
                temp_val = float(questionary.text("LLM Temperature:", default="0.1").ask())
                force_val = questionary.confirm("Force re-analyze already processed items?", default=False).ask()
                
            print("Starting Stage 1 Analysis (Feature Extraction)...")
            run_stage_1_analysis(model_name=model_val, limit=limit_val, temperature=temp_val, force_reanalyze=force_val)
            
        elif choice.startswith("3"):
            # Default or Advanced Stage 2
            embed_val = "sentence-transformers/all-minilm-l12-v2"
            label_val = "google/gemma-4-26b-a4b-it"
            min_size_val = 5
            force_embed_val = False
            force_cluster_val = False
            sample_val = 10
            
            adv = questionary.confirm("Configure advanced Stage 2 options?", default=False).ask()
            if adv:
                embed_val = questionary.text("Embedding Model:", default=embed_val).ask()
                label_val = questionary.text("Labeling Model:", default=label_val).ask()
                min_size_val = int(questionary.text("HDBSCAN min_cluster_size:", default=str(min_size_val)).ask())
                force_embed_val = questionary.confirm("Force re-generate embeddings (bypass cache)?", default=False).ask()
                force_cluster_val = questionary.confirm("Force re-run clustering and LLM labeling?", default=False).ask()
                sample_val = int(questionary.text("Representative items per cluster for labeling:", default=str(sample_val)).ask())
                
            print("Starting Stage 2 Analysis (Thematic Clustering)...")
            run_stage_2_analysis(
                embedding_model=embed_val,
                labeling_model=label_val,
                min_cluster_size=min_size_val,
                force_reembed=force_embed_val,
                force_recluster=force_cluster_val,
                label_sample_size=sample_val
            )
            
        elif choice.startswith("4"):
            print_stats()
            # Show recent records sample
            with Session(engine) as session:
                posts = session.exec(select(Post).limit(10)).all()
                print("\n--- Recent Posts in Database ---")
                if not posts:
                    print("  No posts in database.")
                for p in posts:
                    ann = session.get(Annotation, p.id)
                    tag_str = f"'{ann.consolidated_tag}' (raw: '{ann.raw_tag}')" if (ann and ann.consolidated_tag) else (f"raw: '{ann.raw_tag}'" if (ann and ann.raw_tag) else "None")
                    print(f"  [{p.status.upper()}] ID: {p.id} | Title: {p.title[:50]}... | Theme: {tag_str}")
                    
                comments = session.exec(select(Comment).limit(10)).all()
                print("\n--- Recent Comments in Database ---")
                if not comments:
                    print("  No comments in database.")
                for c in comments:
                    ann = session.get(Annotation, c.id)
                    tag_str = f"'{ann.consolidated_tag}' (raw: '{ann.raw_tag}')" if (ann and ann.consolidated_tag) else (f"raw: '{ann.raw_tag}'" if (ann and ann.raw_tag) else "None")
                    print(f"  [{c.status.upper()}] ID: {c.id} | Body: {c.body[:50]}... | Theme: {tag_str}")
            print()
            
        elif choice.startswith("5"):
            run_db_management_submenu()
            
        elif choice.startswith("6"):
            fmt = questionary.select("Select export format:", choices=["csv", "json"]).ask()
            if not fmt:
                continue
                
            default_out = "export.csv" if fmt == "csv" else "export.json"
            out_path = questionary.text("Enter output file path:", default=default_out).ask()
            if not out_path:
                continue
                
            if fmt == "csv":
                export_to_csv(out_path)
            else:
                export_to_json(out_path)
                
        elif choice.startswith("7"):
            run_visualization_submenu()

def run_db_management_submenu():
    while True:
        manage_choice = questionary.select(
            "Database Management Options:",
            choices=[
                "1. Reset Failed posts/comments to Pending (retry failed)",
                "2. Reset Processed posts to Pending (re-run Stage 1 on posts)",
                "3. Reset Processed comments to Pending (re-run Stage 1 on comments)",
                "4. Wipe Stage 2 Clustering (clear consolidated tags & cluster assignments)",
                "5. Reset specific posts/comments by ID",
                "6. Nuke Database (delete ALL records!)",
                "7. Go Back"
            ]
        ).ask()
        
        if not manage_choice or manage_choice.startswith("7"):
            break
            
        with Session(engine) as session:
            if manage_choice.startswith("1"):
                reset_failed_items(session, "post")
                reset_failed_items(session, "comment")
                print("Failed posts and comments reset to 'pending'.")
                
            elif manage_choice.startswith("2"):
                reset_processed_items(session, "post")
                print("Processed posts reset to 'pending' (corresponding annotations removed).")
                
            elif manage_choice.startswith("3"):
                reset_processed_items(session, "comment")
                print("Processed comments reset to 'pending' (corresponding annotations removed).")
                
            elif manage_choice.startswith("4"):
                clear_stage_2_clustering(session)
                print("Stage 2 clustering data cleared on all annotations.")
                
            elif manage_choice.startswith("5"):
                ids_str = questionary.text("Enter comma-separated IDs to reset (e.g. post1, comment1):").ask()
                if ids_str:
                    item_ids = [i.strip() for i in ids_str.split(",") if i.strip()]
                    reset_specific_items(session, item_ids)
                    print(f"Specified items reset to 'pending' (if present).")
                    
            elif manage_choice.startswith("6"):
                confirm = questionary.confirm("Are you absolutely sure you want to nuke the database? This cannot be undone!", default=False).ask()
                if confirm:
                    nuke_database(session)
                    print("Database nuked. All records deleted.")

def run_visualization_submenu():
    while True:
        vis_choice = questionary.select(
            "Visualization Dashboard Options:",
            choices=[
                "1. Semantic Map (2D t-SNE Plot of Embeddings)",
                "2. Sentiment Distribution by Theme (Box Plot)",
                "3. Theme Trends Over Time (Line Plot)",
                "4. Subreddit Theme Distribution (Stacked Bar Plot)",
                "5. Go Back"
            ]
        ).ask()
        
        if not vis_choice or vis_choice.startswith("5"):
            break
            
        action = questionary.select(
            "What would you like to do with this plot?",
            choices=[
                "1. Preview Plot (Display in window)",
                "2. Export Plot (Save to image file)",
                "3. Go Back"
            ]
        ).ask()
        
        if not action or action.startswith("3"):
            continue
            
        save_path = None
        if action.startswith("2"):
            default_map = {
                "1": "plots/semantic_map.png",
                "2": "plots/sentiment_by_theme.png",
                "3": "plots/theme_trends.png",
                "4": "plots/subreddit_themes.png"
            }
            default_path = default_map.get(vis_choice[0], "plots/plot.png")
            save_path = questionary.text("Enter output image path:", default=default_path).ask()
            if not save_path:
                continue
                
        with Session(engine) as session:
            try:
                if vis_choice.startswith("1"):
                    plot_semantic_map(session, save_path=save_path)
                elif vis_choice.startswith("2"):
                    plot_sentiment_by_theme(session, save_path=save_path)
                elif vis_choice.startswith("3"):
                    bin_choice = questionary.select("Bin time by:", choices=["D (Day)", "W (Week)", "M (Month)"], default="W").ask()
                    bin_choice = bin_choice[0] if bin_choice else "W"
                    plot_theme_trends_over_time(session, save_path=save_path, bin_by=bin_choice)
                elif vis_choice.startswith("4"):
                    plot_subreddit_theme_distribution(session, save_path=save_path)
            except Exception as e:
                print(f"Error generating plot: {e}")

if __name__ == "__main__":
    main()
