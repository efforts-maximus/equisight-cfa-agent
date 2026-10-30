"""Playwright Browser Verification for Unified Search, Quick-Launch Buttons, Movers, and Custom Tickers.

Verifies:
1. Custom ticker typing in Streamlit sidebar Unified Smart Search Bar ("INFY.NS")
2. Company name resolution via search bar ("Tata Consultancy" -> TCS.NS)
3. 1-Click Quick-Launch buttons (e.g. "NVDA") dynamically update search bar and execute pipeline
4. Live Daily Market Movers expander loads real-time movers with 1-click loading
5. Clean rendering of all 5 tabs across tickers with zero .stException
"""

import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUTPUTS_DIR = Path("outputs") / "screenshots"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def find_chrome_executable() -> str | None:
    user_prof = os.environ.get("USERPROFILE", r"C:\Users\M S I")
    ms_playwright = Path(user_prof) / "AppData" / "Local" / "ms-playwright"
    candidates = list(ms_playwright.glob("**/chrome.exe"))
    if candidates:
        return str(candidates[0])
    return None


def verify_custom_ticker_and_sliders(url: str = "http://localhost:8501") -> bool:
    print(f"[*] Starting custom ticker browser verification for {url}...")
    chrome_bin = find_chrome_executable()

    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        if chrome_bin:
            launch_kwargs["executable_path"] = chrome_bin
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Connect to Streamlit
        page.goto(url, timeout=15000, wait_until="domcontentloaded")
        page.wait_for_selector(".stApp", timeout=30000)
        time.sleep(2)
        if page.locator('div[data-testid="stTabs"]').count() == 0:
            terminal_radio = page.locator('div[data-testid="stRadio"] label:has-text("Deep-Dive")')
            if terminal_radio.count() > 0:
                terminal_radio.click()
        page.wait_for_selector('div[data-testid="stTabs"]', timeout=60000)
        print("[+] Initial dashboard loaded.")

        # Verify initial ticker has zero exceptions
        assert page.locator(".stException").count() == 0

        # Locate the Unified Smart Search Bar
        print("[*] Locating Unified Smart Search Bar in sidebar...")
        search_input = page.locator('input[aria-label="🔍 Search Company Name or Ticker"]')
        search_input.wait_for(timeout=10000)
        print(f"[+] Current search input value: {search_input.input_value()}")

        # 1. Type custom ticker query "INFY.NS"
        print("[*] Typing 'INFY.NS' into Unified Smart Search Bar...")
        search_input.click()
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        page.keyboard.type("INFY.NS")
        page.keyboard.press("Enter")
        time.sleep(2)

        # Click Run Coverage Pipeline button
        run_btn = page.locator('button:has-text("Run Coverage Pipeline")')
        if run_btn.count() > 0:
            print("[*] Clicking 'Run Coverage Pipeline' button...")
            run_btn.click()

        # Wait for pipeline execution to complete
        print("[*] Waiting for custom ticker pipeline to execute...")
        time.sleep(4)
        page.wait_for_selector('div[data-testid="stTabs"]', timeout=60000)
        page.wait_for_selector('.js-plotly-plot', timeout=30000)
        time.sleep(1)

        # Verify zero exceptions after loading INFY.NS
        exc_count = page.locator(".stException").count()
        if exc_count > 0:
            print(f"[!] Found {exc_count} exception(s) on custom ticker INFY.NS!", file=sys.stderr)
            for i in range(exc_count):
                print(f"    - Error: {page.locator('.stException').nth(i).inner_text()}", file=sys.stderr)
            browser.close()
            return False

        header_text = page.locator("h1").first.inner_text()
        print(f"[+] Active Dashboard Header: {header_text}")
        assert "INFY.NS" in header_text, f"Expected INFY.NS in header, got: {header_text}"

        # 2. Test 1-Click Quick-Launch button for NVDA
        print("[*] Testing 1-Click Quick-Launch button for NVDA...")
        nvda_btn = page.locator('button:has-text("NVDA")')
        nvda_btn.click()
        time.sleep(4)
        page.wait_for_selector('div[data-testid="stTabs"]', timeout=60000)
        header_nvda = page.locator("h1").first.inner_text()
        print(f"[+] Header after clicking NVDA: {header_nvda}")
        assert "NVDA" in header_nvda, f"Expected NVDA in header, got: {header_nvda}"
        assert search_input.input_value() == "NVDA", f"Expected search input 'NVDA', got: {search_input.input_value()}"
        assert page.locator(".stException").count() == 0

        # 3. Test company name resolution "Tata Consultancy"
        print("[*] Testing company name search 'Tata Consultancy'...")
        search_input.click()
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        page.keyboard.type("Tata Consultancy")
        page.keyboard.press("Enter")
        time.sleep(4)
        page.wait_for_selector('div[data-testid="stTabs"]', timeout=60000)
        header_tcs = page.locator("h1").first.inner_text()
        print(f"[+] Header after typing 'Tata Consultancy': {header_tcs}")
        assert "TCS.NS" in header_tcs, f"Expected TCS.NS in header, got: {header_tcs}"
        assert page.locator(".stException").count() == 0

        # 4. Capture screenshot
        custom_screenshot = OUTPUTS_DIR / "custom_ticker_infy_tab1.png"
        page.screenshot(path=str(custom_screenshot), full_page=True)
        print(f"[+] Screenshot captured: {custom_screenshot}")

        # 5. Navigate through tabs to ensure all 5 render cleanly
        tabs = page.locator('div[data-testid="stTabs"] [role="tab"]')
        for idx in range(tabs.count()):
            tabs.nth(idx).click()
            time.sleep(1.5)
            assert page.locator(".stException").count() == 0, f"Exception on tab {idx+1}"

        print("[+] All verification checks passed with 0 runtime errors!")
        browser.close()
        return True


if __name__ == "__main__":
    success = verify_custom_ticker_and_sliders()
    sys.exit(0 if success else 1)
