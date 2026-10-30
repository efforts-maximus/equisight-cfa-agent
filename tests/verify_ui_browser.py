"""Automated Playwright Browser Verification Script for EquiSight CFA Dashboard.

Launches headless Chromium, visits http://localhost:8501, verifies zero .stException
elements exist across all 5 dashboard tabs, and captures screenshots to outputs/screenshots/.
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


def verify_dashboard_browser(url: str = "http://localhost:8501", max_retries: int = 15) -> bool:
    """Visits the Streamlit application and asserts zero exceptions across Home and all 5 Deep-Dive tabs."""
    print(f"[*] Starting Playwright browser verification for {url}...")
    chrome_bin = find_chrome_executable()
    if chrome_bin:
        print(f"[+] Found global Chromium binary at: {chrome_bin}")

    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        if chrome_bin:
            launch_kwargs["executable_path"] = chrome_bin
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Connect with retry while server starts up
        connected = False
        for attempt in range(1, max_retries + 1):
            try:
                page.goto(url, timeout=10000, wait_until="domcontentloaded")
                connected = True
                print(f"[+] Successfully connected to {url} (attempt {attempt})")
                break
            except Exception as e:
                print(f"[-] Waiting for Streamlit server to be ready (attempt {attempt}/{max_retries})...")
                time.sleep(2)

        if not connected:
            print(f"[!] Could not connect to Streamlit server at {url}", file=sys.stderr)
            browser.close()
            return False

        # Wait for Streamlit app container to render
        page.wait_for_selector(".stApp", timeout=30000)
        time.sleep(3)  # Allow initial UI rendering to settle

        # Check for initial exceptions
        exceptions = page.locator(".stException")
        if exceptions.count() > 0:
            print(f"[!] Detected {exceptions.count()} .stException on initial load!", file=sys.stderr)
            for i in range(exceptions.count()):
                print(f"    - Error {i+1}: {exceptions.nth(i).inner_text()}", file=sys.stderr)
            browser.close()
            return False

        # 1. Verify 🏠 Command Center (Home) Landing Page
        print("[*] Verifying Command Center (Home) Landing Page...")
        page.wait_for_selector('button:has-text("MCap")', timeout=30000)
        time.sleep(2)

        # Capture Command Center Home screenshot
        home_screenshot = OUTPUTS_DIR / "home_command_center.png"
        page.screenshot(path=str(home_screenshot), full_page=True)
        print(f"[+] Screenshot captured: {home_screenshot}")

        # Assert zero exceptions on Home page
        exc_count = page.locator(".stException").count()
        if exc_count > 0:
            print(f"[!] Found {exc_count} exception(s) on Command Center Home!", file=sys.stderr)
            browser.close()
            return False

        # 2. Switch to Deep-Dive Research Terminal via clicking a Live Market Mover Card
        print("[*] Clicking Live Market Mover card to enter Deep-Dive Research Terminal...")
        mover_cards = page.locator('button:has-text("MCap")')
        if mover_cards.count() > 0:
            mover_cards.first.click()
        else:
            terminal_radio = page.locator('div[data-testid="stRadio"] label:has-text("Deep-Dive")')
            terminal_radio.click()

        print("[*] Waiting for Deep-Dive Research Terminal tabs and Plotly charts to render...")
        page.wait_for_selector('div[data-testid="stTabs"]', timeout=60000)
        page.wait_for_selector('.js-plotly-plot', timeout=30000)
        time.sleep(3)

        # Tab definitions and screenshot file targets
        tabs_to_verify = [
            ("Tab 1", "Executive Summary & Football Field", OUTPUTS_DIR / "tab1_football_field.png"),
            ("Tab 2", "2027 CFA FSA & Forensic Quality", OUTPUTS_DIR / "tab2_fsa_dupont.png"),
            ("Tab 3", "DCF Scenarios & Sensitivity", OUTPUTS_DIR / "tab3_dcf_sensitivity.png"),
            ("Tab 4", "Quantitative Risk & Monte Carlo", OUTPUTS_DIR / "tab4_monte_carlo.png"),
            ("Tab 5", "Dialectical Debate, Report & Audit", OUTPUTS_DIR / "tab5_audit_report.png"),
        ]

        # Locate tab elements in Streamlit
        tabs = page.locator('div[data-testid="stTabs"] [role="tab"]')
        tab_count = tabs.count()
        print(f"[+] Located {tab_count} dashboard tab elements.")

        for idx, (tab_prefix, tab_name, screenshot_path) in enumerate(tabs_to_verify):
            print(f"[*] Navigating to {tab_prefix}: {tab_name}...")
            if idx < tab_count:
                tabs.nth(idx).click()
                time.sleep(2.5)  # Allow DOM to render and Plotly charts to draw

            # Verify zero exceptions exist on the active tab
            exc_count = page.locator(".stException").count()
            if exc_count > 0:
                print(f"[!] Found {exc_count} exception(s) on {tab_prefix}!", file=sys.stderr)
                browser.close()
                return False

            # Capture screenshot with full page content
            time.sleep(1.0)
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"[+] Screenshot captured: {screenshot_path}")

        print("[+] Command Center Home and all 5 Streamlit tabs verified successfully with 0 runtime errors.")
        browser.close()
        return True


if __name__ == "__main__":
    port = int(os.environ.get("STREAMLIT_PORT", "8501"))
    target_url = f"http://localhost:{port}"
    success = verify_dashboard_browser(target_url)
    sys.exit(0 if success else 1)
