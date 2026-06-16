import json
import time
from playwright.sync_api import sync_playwright

def test_reddit_json():
    # Example URL for testing JSON endpoint
    url = "https://www.reddit.com/r/python/search.json?q=playwright&limit=3"
    print(f"Target URL: {url}")
    
    with sync_playwright() as p:
        print("Launching Chromium browser...")
        # We run headless=True. If needed, headless=False can be used to debug locally.
        browser = p.chromium.launch(headless=True)
        
        # Define a realistic user agent and viewport to look like a standard browser
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        
        page = context.new_page()
        
        print("\n--- Approach 1: Direct Page Navigation to JSON URL ---")
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=20000)
            status_code = response.status if response else "Unknown"
            print(f"Response Status: {status_code}")
            
            if response and response.ok:
                # Try getting JSON directly from the response
                try:
                    data = response.json()
                    print("SUCCESS (Approach 1 - response.json()): Successfully retrieved JSON!")
                    print(json.dumps(data, indent=2)[:400] + "\n...")
                    browser.close()
                    return
                except Exception as je:
                    print(f"Could not parse response directly as JSON: {je}")
            
            # Browser might render the JSON as text inside a <pre> or body. Let's try evaluating body innerText.
            body_text = page.evaluate("() => document.body.innerText")
            try:
                data = json.loads(body_text)
                print("SUCCESS (Approach 1 - innerText parse): Parsed body text as JSON!")
                print(json.dumps(data, indent=2)[:400] + "\n...")
                browser.close()
                return
            except Exception as je:
                print(f"Could not parse body text as JSON: {je}")
                
        except Exception as e:
            print(f"Approach 1 failed: {e}")
            
        print("\n--- Approach 2: Load Reddit Homepage, then Fetch from Context ---")
        try:
            # First navigate to the main Reddit page so cookies/session headers are established
            print("Navigating to https://www.reddit.com to set cookies...")
            page.goto("https://www.reddit.com", wait_until="networkidle", timeout=30000)
            time.sleep(2) # Give it a moment to run any background scripts
            
            print(f"Executing fetch('{url}') inside the browser context...")
            # Run fetch from within the browser console context (inheriting all cookies/session)
            json_str = page.evaluate(f"""
                async () => {{
                    const response = await fetch('{url}');
                    if (!response.ok) {{
                        throw new Error('HTTP status ' + response.status);
                    }}
                    return await response.text();
                }}
            """)
            
            data = json.loads(json_str)
            print("SUCCESS (Approach 2): Successfully retrieved JSON via in-browser fetch!")
            print(json.dumps(data, indent=2)[:400] + "\n...")
            
        except Exception as e:
            print(f"Approach 2 failed: {e}")
            
        browser.close()

if __name__ == "__main__":
    test_reddit_json()
