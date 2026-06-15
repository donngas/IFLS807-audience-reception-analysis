import os
import sys
import argparse
from dotenv import load_dotenv
import questionary

from database import init_db
from scraper import scrape_reddit
from analyzer import run_stage_1_analysis, run_stage_2_analysis
from util import export_to_csv, export_to_json, print_stats

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
        scrape_reddit(args.query, sub_list, args.post_limit, args.comment_limit)
        
    elif args.action == "analyze":
        print("Starting Stage 1 Analysis (Feature Extraction)...")
        run_stage_1_analysis()
        
    elif args.action == "cluster":
        print("Starting Stage 2 Analysis (Thematic Clustering)...")
        run_stage_2_analysis()
        
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
                "3. Run Stage 2 Analysis (Thematic Clustering)",
                "4. Export data to CSV/JSON",
                "5. Check Pipeline Statistics",
                "6. Exit"
            ]
        ).ask()
        
        if not choice or choice.startswith("6"):
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
            
            scrape_reddit(query, sub_list, int(p_limit), int(c_limit))
            
        elif choice.startswith("2"):
            print("Starting Stage 1 Analysis (Feature Extraction)...")
            run_stage_1_analysis()
            
        elif choice.startswith("3"):
            print("Starting Stage 2 Analysis (Thematic Clustering)...")
            run_stage_2_analysis()
            
        elif choice.startswith("4"):
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
                
        elif choice.startswith("5"):
            print_stats()

if __name__ == "__main__":
    main()
