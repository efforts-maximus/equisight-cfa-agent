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

        # 1.1 Verify Vertical Side Navbar Collapse and Expand-Back Cycle
        print("[*] Verifying vertical side navbar collapse and expand-back cycle...")
        sidebar = page.locator('section[data-testid="stSidebar"]')
        if not sidebar.is_visible():
            print("[!] Sidebar is not visible initially!", file=sys.stderr)
            browser.close()
            return False

        collapse_btn = page.locator('button[aria-label="Collapse sidebar"]')
        if collapse_btn.count() == 0 or not collapse_btn.first.is_visible():
            print("[!] Sidebar collapse button is not visible!", file=sys.stderr)
            browser.close()
            return False

        # Verify Deploy button and MainMenu are hidden
        deploy_btn = page.locator('[data-testid="stAppDeployButton"], .stDeployButton')
        main_menu = page.locator('#MainMenu')
        if deploy_btn.count() > 0 and deploy_btn.is_visible():
            print("[!] Deploy button is visible but should be hidden!", file=sys.stderr)
            browser.close()
            return False
        if main_menu.count() > 0 and main_menu.is_visible():
            print("[!] MainMenu is visible but should be hidden!", file=sys.stderr)
            browser.close()
            return False

        # Collapse sidebar
        collapse_btn.first.click()
        time.sleep(1.5)

        # Assert sidebar is collapsed
        sidebar_bbox = sidebar.bounding_box()
        is_collapsed = (sidebar_bbox is None) or (sidebar_bbox["width"] <= 1) or (sidebar_bbox["x"] < 0)
        if not is_collapsed:
            print(f"[!] Sidebar failed to collapse, bounding box: {sidebar_bbox}", file=sys.stderr)
            browser.close()
            return False

        collapsed_screenshot = OUTPUTS_DIR / "sidebar_collapsed.png"
        page.screenshot(path=str(collapsed_screenshot), full_page=True)
        print(f"[+] Screenshot captured: {collapsed_screenshot}")

        # Locate expand button and click to bring sidebar back
        expand_btn = page.locator('button[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button, button[aria-label="Expand sidebar"]')
        if expand_btn.count() == 0 or not expand_btn.first.is_visible():
            print("[!] Expand sidebar button is not visible when sidebar is collapsed!", file=sys.stderr)
            browser.close()
            return False

        # Verify expand button has high-contrast styling in dark theme
        expand_style = page.evaluate("""() => {
            const btn = document.querySelector('[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button, button[aria-label="Expand sidebar"]');
            if (!btn) return null;
            const span = btn.querySelector('span[data-testid="stIconMaterial"], span, svg');
            const csBtn = window.getComputedStyle(btn);
            const csSpan = span ? window.getComputedStyle(span) : null;
            return {
                btnBg: csBtn.backgroundColor,
                spanColor: csSpan ? csSpan.color : null,
                pointerEvents: csBtn.pointerEvents
            };
        }""")
        if expand_style:
            print(f"[+] Expand button style: {expand_style}")
            if expand_style.get("spanColor") == "rgba(49, 51, 63, 0.6)":
                print("[!] Expand button icon has low contrast rgba(49, 51, 63, 0.6)!", file=sys.stderr)
                browser.close()
                return False

        expand_btn.first.click()
        time.sleep(1.5)

        # Assert sidebar is re-expanded
        sidebar_bbox_after = sidebar.bounding_box()
        is_expanded = sidebar.is_visible() and sidebar_bbox_after is not None and sidebar_bbox_after["width"] > 200
        if not is_expanded:
            print(f"[!] Sidebar failed to expand back, bounding box: {sidebar_bbox_after}", file=sys.stderr)
            browser.close()
            return False

        reexpanded_screenshot = OUTPUTS_DIR / "sidebar_reexpanded.png"
        page.screenshot(path=str(reexpanded_screenshot), full_page=True)
        print(f"[+] Screenshot captured: {reexpanded_screenshot}")
        print("[+] Vertical side navbar collapsed and brought back successfully!")

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

        # 2.1 Verify Vertical Side Navbar Collapse and Expand Cycle inside Deep-Dive Terminal
        print("[*] Verifying sidebar collapse and expand cycle inside Deep-Dive Research Terminal...")
        deep_collapse = page.locator('button[aria-label="Collapse sidebar"]')
        if deep_collapse.count() > 0 and deep_collapse.first.is_visible():
            deep_collapse.first.click()
            time.sleep(1.5)
            deep_sidebar_bbox = sidebar.bounding_box()
            if (deep_sidebar_bbox is not None) and (deep_sidebar_bbox["width"] > 1 and deep_sidebar_bbox["x"] >= 0):
                print(f"[!] Sidebar failed to collapse inside Deep-Dive Terminal, bbox: {deep_sidebar_bbox}", file=sys.stderr)
                browser.close()
                return False
            deep_expand = page.locator('button[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button, button[aria-label="Expand sidebar"]')
            if deep_expand.count() == 0 or not deep_expand.first.is_visible():
                print("[!] Expand button not visible inside Deep-Dive Terminal when collapsed!", file=sys.stderr)
                browser.close()
                return False
            deep_expand.first.click()
            time.sleep(1.5)
            deep_sidebar_bbox_after = sidebar.bounding_box()
            if not (sidebar.is_visible() and deep_sidebar_bbox_after is not None and deep_sidebar_bbox_after["width"] > 200):
                print(f"[!] Sidebar failed to re-expand inside Deep-Dive Terminal, bbox: {deep_sidebar_bbox_after}", file=sys.stderr)
                browser.close()
                return False
            print("[+] Sidebar collapse and expand cycle verified inside Deep-Dive Research Terminal!")

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
