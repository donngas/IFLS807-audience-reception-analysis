import os
import sys
import argparse
from dotenv import load_dotenv
import questionary

def main():
    load_dotenv()
    
    # 1. Determine workspace (database file name) before importing database models or pipeline functions
    if len(sys.argv) == 1:
        # Interactive mode: Select or Create Workspace
        db_files = [f for f in os.listdir(".") if f.endswith(".db") and f not in ["test.db", "audience_reception.db"]]
        if os.path.exists("audience_reception.db"):
            db_files.insert(0, "audience_reception.db")
        else:
            db_files.append("audience_reception.db")
            
        print("=" * 40)
        print("     WORKSPACE & DATABASE SELECTION")
        print("=" * 40)
        choices = []
        for idx, f in enumerate(db_files, 1):
            choices.append(f"{idx}. {f[:-3]} (existing database)")
        choices.append(f"{len(db_files) + 1}. Create a new workspace")
        
        choice = questionary.select(
            "Select or create a workspace (database):",
            choices=choices,
            default=choices[0] if choices else None
        ).ask()
        
        if not choice:
            sys.exit(0)
            
        if "Create a new workspace" in choice:
            name = questionary.text("Enter a name for the new workspace (e.g. jake_amy):").ask()
            if not name:
                name = "default"
            # Sanitize name
            name = "".join([c for c in name if c.isalnum() or c in "_-"]).strip()
            db_file = f"{name}.db"
        else:
            idx = int(choice.split(".")[0]) - 1
            db_file = db_files[idx]
            
        os.environ["WORKSPACE_DB"] = db_file
        print(f"\nActive Workspace: {db_file[:-3].upper()} ({db_file})\n")
    else:
        # CLI mode: Scan arguments for --workspace
        workspace_name = "audience_reception"
        if "--workspace" in sys.argv:
            idx = sys.argv.index("--workspace")
            if idx + 1 < len(sys.argv):
                workspace_name = sys.argv[idx + 1]
        os.environ["WORKSPACE_DB"] = f"{workspace_name}.db"
        print(f"Active Workspace: {workspace_name.upper()} ({workspace_name}.db)\n")

    # 2. Deferred imports: Now that WORKSPACE_DB is configured, we can import pipeline modules
    from database import init_db
    init_db()
    
    from scraper import scrape_reddit
    from analyzer import run_stage_1_analysis, run_stage_2_analysis
    from util import export_to_csv, export_to_json, print_stats

    parser = argparse.ArgumentParser(description="Audience Reception Analysis Pipeline")
    parser.add_argument("--action", choices=["scrape", "analyze", "cluster", "export", "stats"], help="Action to perform")
    parser.add_argument("--workspace", type=str, default="audience_reception", help="Workspace database name")
    
    # Scrape args
    parser.add_argument("--query", type=str, help="Search query for Reddit")
    parser.add_argument("--subreddits", type=str, help="Comma-separated list of subreddits")
    parser.add_argument("--post-limit", type=int, default=100, help="Max posts to scrape")
    parser.add_argument("--comment-limit", type=int, default=100, help="Max comments per post to scrape")
    parser.add_argument("--sort", type=str, default="top", help="Reddit search sort")
    parser.add_argument("--time-filter", type=str, default="all", help="Reddit search time filter")
    parser.add_argument("--force-overwrite", action="store_true", help="Force overwrite existing scraped items")
    parser.add_argument("--method", choices=["praw", "json", "playwright"], default="praw", help="Data acquisition method (praw, json, or playwright)")
    
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
        scrape_reddit(args.query, sub_list, args.post_limit, args.comment_limit, sort=args.sort, time_filter=args.time_filter, skip_existing=not args.force_overwrite, method=args.method)
        
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
    # Local imports to prevent premature SQL engine initialization
    from sqlmodel import Session, select
    from database import engine, Post, Comment, Annotation
    from scraper import scrape_reddit
    from analyzer import run_stage_1_analysis, run_stage_2_analysis
    from util import export_to_csv, export_to_json, print_stats

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
            print("\n=== Reddit Query Setup ===")
            query_method = questionary.select(
                "How would you like to enter your search query?",
                choices=[
                    "1. Enter query directly (for experienced users)",
                    "2. Use Interactive Query Builder (helper wizard)",
                    "3. Read Reddit Search Syntax Guide"
                ]
            ).ask()
            
            if not query_method:
                continue
                
            query = ""
            if query_method.startswith("3"):
                print("\n" + "="*50)
                print("REDDIT SEARCH QUERY SYNTAX GUIDE")
                print("="*50)
                print("Reddit search supports Boolean operators (must be UPPERCASE):")
                print("  - AND: Matches both terms (e.g. 'Jake AND Amy')")
                print("  - OR: Matches either term (e.g. 'chemistry OR romance')")
                print("  - NOT: Excludes terms (e.g. 'chemistry NOT physics')")
                print("Grouping and exact matches:")
                print("  - Parentheses: Group operators (e.g. '(Jake AND Amy) AND (chemistry OR pacing)')")
                print("  - Quotes: Match exact phrase (e.g. '\"pacing issues\"')")
                print("Field searches:")
                print("  - title: Search titles only (e.g. 'title:\"WandaVision\"')")
                print("  - author: Search by author (e.g. 'author:AutoModerator')")
                print("="*50 + "\n")
                
                query = questionary.text("Enter search query (e.g. {Jake AND Amy}):").ask()
            elif query_method.startswith("2"):
                # Interactive Query Builder
                print("\n--- Interactive Query Builder ---")
                primary = questionary.text("Enter primary search terms (must match all, e.g., Jake, Amy):").ask()
                optional = questionary.text("Enter optional terms (matches any of these, e.g., chemistry, romance, pacing - leave blank if none):").ask()
                exclude = questionary.text("Enter terms to exclude (leave blank if none):").ask()
                
                parts = []
                if primary:
                    primary_terms = [t.strip() for t in primary.split(",") if t.strip()]
                    if len(primary_terms) > 1:
                        parts.append("(" + " AND ".join(primary_terms) + ")")
                    else:
                        parts.append(primary_terms[0])
                        
                if optional:
                    optional_terms = [t.strip() for t in optional.split(",") if t.strip()]
                    if len(optional_terms) > 1:
                        parts.append("(" + " OR ".join(optional_terms) + ")")
                    else:
                        parts.append(optional_terms[0])
                        
                exclude_part = ""
                if exclude:
                    exclude_terms = [t.strip() for t in exclude.split(",") if t.strip()]
                    if len(exclude_terms) > 1:
                        exclude_part = "NOT (" + " OR ".join(exclude_terms) + ")"
                    else:
                        exclude_part = f"NOT {exclude_terms[0]}"
                
                query = " AND ".join(parts)
                if exclude_part:
                    query = f"{query} {exclude_part}" if query else exclude_part
                    
                print(f"\nGenerated Query: {query}")
                confirm_query = questionary.confirm("Use this generated query?", default=True).ask()
                if not confirm_query:
                    continue
            else:
                query = questionary.text("Enter search query (e.g. {Jake AND Amy}):").ask()
                
            if not query:
                continue
                
            sub_str = questionary.text("Enter subreddits (comma-separated, leave blank for all):").ask()
            sub_list = [s.strip() for s in sub_str.split(",")] if sub_str else None
            
            p_limit = questionary.text("Post limit:", default="100").ask()
            c_limit = questionary.text("Comment limit per post:", default="100").ask()
            
            # Data Acquisition Method Choice
            method_choice = questionary.select(
                "Select data acquisition method:",
                choices=[
                    "1. PRAW (Reddit API - requires credentials)",
                    "2. Unauthenticated JSON (Bypass API keys)",
                    "3. Playwright (Direct headless browser scraping)"
                ],
                default="1. PRAW (Reddit API - requires credentials)"
            ).ask()
            method_val = "praw" if "PRAW" in method_choice else ("playwright" if "Playwright" in method_choice else "json")
            
            # Advanced Scraping Options
            sort_val = "top"
            time_val = "all"
            skip_val = True
            adv = questionary.confirm("Configure advanced scraping options?", default=False).ask()
            if adv:
                sort_val = questionary.select("Sort by:", choices=["top", "hot", "new", "relevance"], default="top").ask()
                time_val = questionary.select("Time filter:", choices=["all", "day", "week", "month", "year"], default="all").ask()
                skip_val = questionary.confirm("Skip already scraped posts?", default=True).ask()
                
            scrape_reddit(query, sub_list, int(p_limit), int(c_limit), sort=sort_val, time_filter=time_val, skip_existing=skip_val, method=method_val)
            
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
            run_interactive_db_viewer()
            
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
    from sqlmodel import Session
    from database import (
        engine, reset_failed_items, reset_processed_items,
        clear_stage_2_clustering, reset_specific_items, nuke_database
    )

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
    from sqlmodel import Session
    from database import engine
    from visualization import (
        plot_semantic_map, plot_sentiment_by_theme,
        plot_theme_trends_over_time, plot_subreddit_theme_distribution
    )

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

def parse_range_selection(selection_str: str, max_val: int) -> list[int]:
    """Parse selection string like '5, 12-15, 20' into a sorted list of 1-based indices."""
    selected = set()
    parts = [p.strip() for p in selection_str.replace(",", " ").split() if p.strip()]
    for part in parts:
        if "-" in part:
            try:
                start, end = part.split("-")
                start_val = int(start.strip())
                end_val = int(end.strip())
                if start_val <= end_val:
                    for val in range(start_val, end_val + 1):
                        if 1 <= val <= max_val:
                            selected.add(val)
                else:
                    for val in range(end_val, start_val + 1):
                        if 1 <= val <= max_val:
                            selected.add(val)
            except ValueError:
                continue
        else:
            try:
                val = int(part)
                if 1 <= val <= max_val:
                    selected.add(val)
            except ValueError:
                continue
    return sorted(list(selected))

def run_interactive_db_viewer():
    from sqlmodel import Session, select
    from database import engine, Post, Comment, Annotation
    from util import print_stats
    import time
    
    while True:
        choice = questionary.select(
            "Interactive Database Viewer Options:",
            choices=[
                "1. View Posts (paginated list)",
                "2. View Comments (paginated list)",
                "3. View Database Statistics",
                "4. Go Back"
            ]
        ).ask()
        
        if not choice or choice.startswith("4"):
            break
            
        if choice.startswith("3"):
            print_stats()
            continue
            
        item_type = "post" if choice.startswith("1") else "comment"
        model_cls = Post if item_type == "post" else Comment
        
        page = 1
        page_size = 10
        status_filter = "All"
        tag_filter = "All"
        
        while True:
            with Session(engine) as session:
                # Base query
                stmt = select(model_cls)
                if status_filter != "All":
                    stmt = stmt.where(model_cls.status == status_filter.lower())
                
                # Tag filter requires joining with Annotation
                if tag_filter != "All":
                    stmt = stmt.join(Annotation, Annotation.item_id == model_cls.id, isouter=True)
                    if tag_filter == "None":
                        stmt = stmt.where(Annotation.id == None)
                    else:
                        stmt = stmt.where(Annotation.consolidated_tag == tag_filter)
                        
                # Order by created_utc desc
                stmt = stmt.order_by(model_cls.created_utc.desc())
                all_matching_items = session.exec(stmt).all()
                total_count = len(all_matching_items)
                
                total_pages = max(1, (total_count + page_size - 1) // page_size)
                if page > total_pages:
                    page = total_pages
                
                # Slice items for current page
                items_page = all_matching_items[(page - 1) * page_size : page * page_size]
                
                border_color = "\033[94m" # Blue
                reset_color = "\033[0m"
                header_color = "\033[96m\033[1m" # Bold Cyan
                text_color = "\033[37m" # Light Gray
                theme_color = "\033[95m" # Magenta
                pipe_color = "\033[90m" # Dark Gray
                
                print(f"\n{border_color}" + "=" * 110 + f"{reset_color}")
                print(f"{header_color} VIEWING {item_type.upper()}S (Page {page}/{total_pages} | Total Matching: {total_count}){reset_color}")
                print(f"{text_color} Status Filter: {status_filter} | Theme Filter: {tag_filter}{reset_color}")
                print(f"{border_color}" + "=" * 110 + f"{reset_color}")
                
                if not items_page:
                    print("  No records found matching current filters.")
                else:
                    # Print table column headers
                    print(f"  {header_color}{'#':<3}{reset_color} {pipe_color}|{reset_color} {header_color}{'STATUS':<11}{reset_color} {pipe_color}|{reset_color} {header_color}{'ID':<10}{reset_color} {pipe_color}|{reset_color} {header_color}{'TEXT SNIPPET':<45}{reset_color} {pipe_color}|{reset_color} {header_color}{'THEME/TAG'}{reset_color}")
                    print(f"{border_color}" + "-" * 110 + f"{reset_color}")
                    
                    for i, item in enumerate(items_page):
                        # superficial global integer index (1-based globally)
                        global_idx = (page - 1) * page_size + i + 1
                        
                        ann = session.get(Annotation, item.id)
                        tag_str = "None"
                        if ann:
                            tag_str = f"'{ann.consolidated_tag}'" if ann.consolidated_tag else f"raw: '{ann.raw_tag}'"
                        text_snippet = item.title if item_type == "post" else item.body
                        text_snippet = text_snippet.replace('\n', ' ')[:45] + "..." if len(text_snippet) > 45 else text_snippet.replace('\n', ' ')
                        
                        # Pad the status string before adding ANSI escape codes
                        status_raw = f"[{item.status.upper()}]"
                        status_padded = f"{status_raw:<11}"
                        if "PENDING" in status_padded:
                            status_str = status_padded.replace("[PENDING]", "\033[93m[PENDING]\033[0m") # Yellow
                        elif "PROCESSED" in status_padded:
                            status_str = status_padded.replace("[PROCESSED]", "\033[92m[PROCESSED]\033[0m") # Green
                        elif "FAILED" in status_padded:
                            status_str = status_padded.replace("[FAILED]", "\033[91m[FAILED]\033[0m") # Red
                        elif "SKIPPED" in status_padded:
                            status_str = status_padded.replace("[SKIPPED]", "\033[94m[SKIPPED]\033[0m") # Blue
                        else:
                            status_str = status_padded
                            
                        id_padded = f"{item.id:<10}"
                        snippet_padded = f"{text_snippet:<45}"
                        
                        print(f"  {global_idx:3d}. {pipe_color}|{reset_color} {status_str} {pipe_color}|{reset_color} {id_padded} {pipe_color}|{reset_color} {snippet_padded} {pipe_color}|{reset_color} {theme_color}{tag_str}{reset_color}")
                
                print(f"{border_color}" + "=" * 110 + f"{reset_color}")
                
                # Build typical one-liner Linux CLI menu guide
                allowed_actions = []
                action_options = []
                
                nav_color = "\033[93m" # Light Yellow
                menu_btn_color = "\033[95m" # Magenta
                
                if page < total_pages:
                    action_options.append(f"{nav_color}[N]ext Page{reset_color}")
                    allowed_actions.append("n")
                if page > 1:
                    action_options.append(f"{nav_color}[P]rev Page{reset_color}")
                    allowed_actions.append("p")
                
                if items_page:
                    action_options.append(f"{menu_btn_color}[S]elect/Manipulate{reset_color}")
                    allowed_actions.append("s")
                    
                action_options.extend([
                    f"{nav_color}[F]ilter Status{reset_color}",
                    f"{nav_color}[T]heme Filter{reset_color}",
                    f"{nav_color}[H]elp/Legend{reset_color}",
                    f"{nav_color}[B]ack{reset_color}"
                ])
                allowed_actions.extend(["f", "t", "h", "b"])
                
                print(f"\033[90mActions:\033[0m " + f" {pipe_color}|{reset_color} ".join(action_options))
                action_char = questionary.text("Choose action:").ask()
                
                if not action_char:
                    continue
                
                action_char = action_char.strip().lower()
                if action_char not in allowed_actions:
                    print("Invalid option or action not available on this page.")
                    time.sleep(1)
                    continue
                
                if action_char == "b":
                    break
                elif action_char == "n":
                    page += 1
                elif action_char == "p":
                    page -= 1
                elif action_char == "f":
                    status_filter = questionary.select(
                        "Select status filter:",
                        choices=["All", "Pending", "Processed", "Failed", "Skipped"]
                    ).ask() or "All"
                    page = 1
                elif action_char == "t":
                    # Fetch all unique consolidated tags for filtering
                    tags_stmt = select(Annotation.consolidated_tag).where(Annotation.item_type == item_type).distinct()
                    unique_tags = [t for t in session.exec(tags_stmt).all() if t]
                    filter_choices = ["All", "None"] + unique_tags
                    tag_filter = questionary.select(
                        "Select theme/tag filter:",
                        choices=filter_choices
                    ).ask() or "All"
                    page = 1
                elif action_char == "h":
                    print("\n" + "="*50)
                    print("DATABASE VIEWER HELP & LEGEND")
                    print("="*50)
                    print("Statuses:")
                    print("  - PENDING:   Scraped successfully, awaiting Stage 1 analysis.")
                    print("  - PROCESSED: Stage 1 analysis completed (annotated with sentiment, tag, etc.).")
                    print("  - FAILED:    Stage 1 analysis failed (e.g. due to API timeout or error).")
                    print("  - SKIPPED:   Exposed as low quality during scraping and excluded.")
                    print("\nWhy are some rows marked as SKIPPED?")
                    print("  An item acquires SKIPPED status during scraping if:")
                    print("  1. The author is 'AutoModerator' or contains 'bot' (filtered out bots).")
                    print("  2. The text body is '[deleted]', '[removed]', or completely empty.")
                    print("="*50 + "\n")
                    questionary.text("Press Enter to continue...").ask()
                elif action_char == "s":
                    # Prompt for selection by superficial global index number
                    selection_str = questionary.text(
                        "Enter row number, comma-separated list, or range to manipulate (e.g., 5, 12-15, 20):"
                    ).ask()
                    if not selection_str:
                        continue
                        
                    parsed_idx = parse_range_selection(selection_str, total_count)
                    if not parsed_idx:
                        print("No valid row numbers entered.")
                        time.sleep(1)
                        continue
                        
                    # Retrieve the selected items using the parsed global indices
                    selected_items = [all_matching_items[i - 1] for i in parsed_idx]
                    
                    if len(selected_items) == 1:
                        manipulate_item(session, selected_items[0], item_type)
                    else:
                        manipulate_items_bulk(session, selected_items, item_type)

def manipulate_item(session, item, item_type):
    from database import Annotation, upsert_post, upsert_comment, upsert_annotation
    
    session.add(item)
    
    while True:
        ann = session.get(Annotation, item.id)
        print("\n==========================================")
        print(f" ITEM DETAILS & MANIPULATION")
        print(f"==========================================")
        print(f"Type:        {item_type.upper()}")
        print(f"ID:          {item.id}")
        print(f"Subreddit:   {item.subreddit if item_type == 'post' else 'N/A'}")
        print(f"Status:      {item.status.upper()}")
        print(f"Score:       {item.score}")
        print(f"Created UTC: {item.created_utc}")
        if item_type == "post":
            print(f"Title:       {item.title}")
            print(f"Selftext:\n{item.selftext}")
        else:
            print(f"Body:\n{item.body}")
            
        print("------------------------------------------")
        print("Annotation Details:")
        if ann:
            print(f"  Sentiment:        {ann.sentiment}")
            print(f"  Raw Tag:          {ann.raw_tag}")
            print(f"  Consolidated Tag: {ann.consolidated_tag}")
            print(f"  Cluster ID:       {ann.cluster_id}")
            print(f"  Summary:          {ann.summary}")
        else:
            print("  (No annotation available)")
        print("==========================================")
        
        manip_choice = questionary.select(
            "What would you like to do with this item?",
            choices=[
                "1. Reset Status to Pending (deletes annotation)",
                "2. Edit Sentiment & Tags Manually",
                "3. Delete Item completely",
                "4. Go Back to List"
            ]
        ).ask()
        
        if not manip_choice or manip_choice.startswith("4"):
            break
            
        if manip_choice.startswith("1"):
            item.status = "pending"
            if item_type == "post":
                upsert_post(session, item)
            else:
                upsert_comment(session, item)
            if ann:
                session.delete(ann)
            session.commit()
            print("Item reset to 'pending' and annotation removed.")
            break
            
        elif manip_choice.startswith("2"):
            new_sentiment = float(questionary.select(
                "Select sentiment polarity score:",
                choices=["-1.0", "-0.5", "0.0", "0.5", "1.0"],
                default=str(ann.sentiment) if ann else "0.0"
            ).ask() or 0.0)
            new_raw_tag = questionary.text("Enter Raw Tag:", default=ann.raw_tag if ann else "").ask() or "Reaction Only"
            new_consolidated = questionary.text("Enter Consolidated Tag (Theme):", default=ann.consolidated_tag if ann else "").ask() or ""
            
            if not ann:
                ann = Annotation(
                    item_id=item.id,
                    item_type=item_type,
                    sentiment=new_sentiment,
                    raw_tag=new_raw_tag,
                    consolidated_tag=new_consolidated or None,
                    summary="Manually annotated"
                )
            else:
                ann.sentiment = new_sentiment
                ann.raw_tag = new_raw_tag
                ann.consolidated_tag = new_consolidated or None
                ann.summary = ann.summary or "Manually edited"
                
            upsert_annotation(session, ann)
            item.status = "processed"
            if item_type == "post":
                upsert_post(session, item)
            else:
                upsert_comment(session, item)
            session.commit()
            print("Item manual annotation updated successfully.")
            
        elif manip_choice.startswith("3"):
            confirm = questionary.confirm("Are you sure you want to delete this item? This will also remove any annotations.", default=False).ask()
            if confirm:
                if ann:
                    session.delete(ann)
                session.delete(item)
                session.commit()
                print("Item deleted from database.")
                break

def manipulate_items_bulk(session, items, item_type):
    from database import Annotation, upsert_post, upsert_comment, upsert_annotation
    
    print("\n==========================================")
    print(f" BULK MANIPULATION ({len(items)} items selected)")
    print("==========================================")
    
    # Show brief list of selected item IDs
    ids_str = ", ".join([item.id for item in items[:10]])
    if len(items) > 10:
        ids_str += f"... and {len(items) - 10} more"
    print(f"Selected IDs: {ids_str}")
    
    choice = questionary.select(
        "What would you like to do with these items in bulk?",
        choices=[
            "1. Reset Status to Pending (deletes annotations)",
            "2. Set Sentiment & Tags in bulk",
            "3. Delete items completely",
            "4. Cancel"
        ]
    ).ask()
    
    if not choice or choice.startswith("4"):
        return
        
    if choice.startswith("1"):
        for item in items:
            session.add(item)
            item.status = "pending"
            if item_type == "post":
                upsert_post(session, item)
            else:
                upsert_comment(session, item)
            ann = session.get(Annotation, item.id)
            if ann:
                session.delete(ann)
        session.commit()
        print(f"Successfully reset {len(items)} items to 'pending' and removed annotations.")
        
    elif choice.startswith("2"):
        new_sentiment = float(questionary.select(
            "Select sentiment polarity score:",
            choices=["-1.0", "-0.5", "0.0", "0.5", "1.0"],
            default="0.0"
        ).ask() or 0.0)
        new_raw_tag = questionary.text("Enter Raw Tag:").ask() or "Reaction Only"
        new_consolidated = questionary.text("Enter Consolidated Tag (Theme - leave blank if none):").ask() or ""
        
        for item in items:
            session.add(item)
            ann = session.get(Annotation, item.id)
            if not ann:
                ann = Annotation(
                    item_id=item.id,
                    item_type=item_type,
                    sentiment=new_sentiment,
                    raw_tag=new_raw_tag,
                    consolidated_tag=new_consolidated or None,
                    summary="Bulk manually annotated"
                )
            else:
                ann.sentiment = new_sentiment
                ann.raw_tag = new_raw_tag
                ann.consolidated_tag = new_consolidated or None
                ann.summary = "Bulk manually edited"
            
            upsert_annotation(session, ann)
            item.status = "processed"
            if item_type == "post":
                upsert_post(session, item)
            else:
                upsert_comment(session, item)
        session.commit()
        print(f"Successfully updated annotations and status for {len(items)} items.")
        
    elif choice.startswith("3"):
        confirm = questionary.confirm(f"Are you absolutely sure you want to delete these {len(items)} items? This cannot be undone!", default=False).ask()
        if confirm:
            for item in items:
                session.add(item)
                ann = session.get(Annotation, item.id)
                if ann:
                    session.delete(ann)
                session.delete(item)
            session.commit()
            print(f"Successfully deleted {len(items)} items from database.")

if __name__ == "__main__":
    main()
