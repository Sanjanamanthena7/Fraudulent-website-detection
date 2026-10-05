import re
from urllib.parse import urlparse
import pandas as pd


def extract_features(url: str) -> pd.DataFrame:
    """Extract lexical and structural features from a given URL."""
    # Ensure scheme exists for parsing
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path

    # 1. Presence of IP address instead of domain name
    ip_pattern = r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])"
    has_ip = 1 if re.search(ip_pattern, hostname) else 0

    # 2. Length of URL
    url_len = len(url)

    # 3. Abnormal characters and count markers
    count_dots = url.count(".")
    count_hyphens = url.count("-")
    count_at = url.count("@")
    count_question = url.count("?")
    count_percent = url.count("%")
    count_equals = url.count("=")

    # 4. HTTPS protocol usage
    is_https = 1 if parsed.scheme.lower() == "https" else 0

    # 5. Suspicious keywords often present in phishing paths
    phish_keywords = ["secure", "account", "update", "banking", "login", "verify", "signin"]
    has_phish_keyword = 1 if any(keyword in url.lower() for keyword in phish_keywords) else 0

    # 6. Subdomain count
    domain_parts = hostname.split(".")
    subdomain_count = max(0, len(domain_parts) - 2)

    features = {
        "url_length": url_len,
        "has_ip": has_ip,
        "count_dots": count_dots,
        "count_hyphens": count_hyphens,
        "count_at": count_at,
        "count_question": count_question,
        "count_percent": count_percent,
        "count_equals": count_equals,
        "is_https": is_https,
        "has_phish_keyword": has_phish_keyword,
        "subdomain_count": subdomain_count,
    }

    return pd.DataFrame([features])