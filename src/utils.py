import re
import os
from urllib.parse import urlparse

# --- Trusted sources list ---
CREDIBLE_DOMAINS = {
    "wikipedia.org", "bbc.com", "reuters.com", "apnews.com", "aljazeera.com",
    "espn.com", "uefa.com", "fifa.com", "theguardian.com", "nytimes.com",
    "indiatoday.in", "timesofindia.indiatimes.com", "ndtv.com"
}

# --- Suspicious phrases for clickbait detection ---
CLICKBAIT_PATTERNS = [
    r"you won't believe", r"shocking", r"miracle", r"exposed", r"instant"
]

# --- Regex for cleaning ---
WHITESPACE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Normalize whitespace and strip text safely."""
    if not s:
        return ""
    s = s.strip()
    s = WHITESPACE.sub(" ", s)
    return s


def domain_of(url: str) -> str:
    """Extract domain (2nd level + TLD) from a URL."""
    try:
        netloc = urlparse(url).netloc.lower()
        # strip subdomains
        parts = netloc.split('.')
        return '.'.join(parts[-2:]) if len(parts) >= 2 else netloc
    except Exception:
        return ""


def is_credible(url: str) -> bool:
    """Check if a URL belongs to a credible domain."""
    d = domain_of(url)
    return any(d.endswith(cd) for cd in CREDIBLE_DOMAINS)


def softmax2(p0: float, p1: float):
    """
    Defensive normalization of 2-class probabilities.
    Returns a tuple (p0_norm, p1_norm).
    """
    s = (p0 or 0.0) + (p1 or 0.0)
    if s <= 0:
        return 0.5, 0.5
    return (p0 / s), (p1 / s)
