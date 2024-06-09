from typing import Annotated
from ae.core.playwright_manager import PlaywrightManager
from ae.utils.logger import logger


async def openurl(url: Annotated[str, "The URL to navigate to. Value must include the protocol (http:// or https://)."],
            timeout: Annotated[int, "Additional wait time in seconds after initial load."] = 6) -> Annotated[str, "Returns the result of this request in text form"]:
    """
    Opens a specified URL in the active browser instance. Waits for an initial load event, then waits for either
    the 'domcontentloaded' event or a configurable timeout, whichever comes first.

    Parameters:
    - url: The URL to navigate to.
    - timeout: Additional time in seconds to wait after the initial load before considering the navigation successful.

    Returns:
    - URL of the new page.
    """
    logger.info(f"Opening URL: {url}")
    print(f"Opening URL: {url}")
    browser_manager = PlaywrightManager(browser_type='chromium', headless=False)
    await browser_manager.get_browser_context()
    page = await browser_manager.get_current_page()
    # Navigate to the URL with a short timeout to ensure the initial load starts
    try:
        url = ensure_protocol(url)
        await page.goto(url, timeout=timeout*1000) # type: ignore
    except Exception as e:
        logger.warn(f"Initial navigation to {url} failed: {e}. Will try to continue anyway.") # happens more often than not, but does not seem to be a problem
        import traceback
        traceback.print_exc()
    await browser_manager.notify_user(f"Opened URL: {url}")
        # Get the page title
    title = await page.title()
    url=page.url
    return f"Page loaded: {url}, Title: {title}" # type: ignore

def ensure_protocol(url: str) -> str:
    """
    Ensures that a URL has a protocol (http:// or https://). If it doesn't have one,
    https:// is added by default.

    Parameters:
    - url: The URL to check and modify if necessary.

    Returns:
    - A URL string with a protocol.
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url  # Default to http if no protocol is specified
        logger.info(f"Added 'https://' protocol to URL because it was missing. New URL is: {url}")
    return url
