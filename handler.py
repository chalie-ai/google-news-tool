"""
Google News Tool Handler — Search Google News for current articles.

Uses the GoogleNews Python package (scrapes Google News, no API key required).
Returns structured article data with two-layer deduplication and importance reranking.
"""

import logging
import re
import time
from typing import Optional
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "are", "was",
    "were", "and", "or", "but", "not", "with", "from", "by", "as", "it", "its",
    "this", "that", "has", "have", "had", "be", "been", "will", "would", "can",
    "could", "may", "might", "do", "does", "did", "up", "out", "over", "after",
    "into", "than", "how", "what", "when", "who", "all", "more", "about",
}


def execute(topic: str, params: dict, config: dict = None, telemetry: dict = None) -> dict:
    """
    Search Google News and return structured article data.

    Args:
        topic: Conversation topic (passed by framework, unused directly)
        params: {
            "query": str (required),
            "limit": int (optional, default 5, clamped to 1-8),
            "region": str (optional, e.g. "US", "GB"),
            "period": str (optional, e.g. "1d", "7d", "30d"),
            "language": str (optional, e.g. "en")
        }
        config: Tool config from DB (unused — no API key needed)
        telemetry: Client telemetry dict with locale, language fields

    Returns:
        {
            "results": [{"title", "source", "date", "published_at", "description",
                         "url", "image_url", "also_reported_by"}],
            "count": int,
            "query": str,
            "_meta": {observability fields}
        }
    """
    query = (params.get("query") or "").strip()
    if not query:
        return {"results": [], "count": 0, "query": "", "_meta": {}}

    limit = max(1, min(8, int(params.get("limit") or 5)))
    region = (params.get("region") or "").strip().upper()
    period = (params.get("period") or "").strip()
    language = (params.get("language") or "").strip().lower()

    # Auto-detect region and language from telemetry when not specified
    if telemetry:
        if not region:
            locale = telemetry.get("locale", "")  # e.g. "en_US"
            if "_" in locale:
                region = locale.split("_")[-1].upper()
        if not language:
            lang_tel = (telemetry.get("language", "") or "").strip()
            if lang_tel:
                language = lang_tel.split("-")[0].lower()

    region = region or "US"
    language = language or "en"

    t0 = time.time()
    raw_articles, retry_used, fetch_error = _fetch_with_retry(query, region, period, language)
    fetch_latency_ms = int((time.time() - t0) * 1000)

    if fetch_error and not raw_articles:
        logger.error(
            '{"event":"fetch_error","query":"%s","error_type":"%s",'
            '"retry_attempted":true,"retry_succeeded":false,"fetch_latency_ms":%d}',
            query, _classify_error(fetch_error), fetch_latency_ms,
        )
        return {"results": [], "count": 0, "query": query, "error": str(fetch_error)[:200], "_meta": {}}

    article_count_raw = len(raw_articles)

    # Two-layer deduplication
    url_deduped = _dedup_by_url(raw_articles)
    title_deduped = _dedup_by_title(url_deduped)
    article_count_deduped = len(title_deduped)
    dedup_removed = article_count_raw - article_count_deduped

    # Importance reranking: multi-source boost (primary) then freshness (secondary)
    ranked = _rerank(title_deduped)

    results = ranked[:limit]

    image_count = sum(1 for r in results if r.get("image_url"))
    image_availability_rate = round(image_count / len(results), 2) if results else 0.0
    source_diversity = len({r.get("source", "") for r in results if r.get("source")})
    dedup_ratio = round(dedup_removed / article_count_raw, 2) if article_count_raw else 0.0

    logger.info(
        '{"event":"fetch_ok","query":"%s","article_count":%d,"image_count":%d,'
        '"dedup_removed":%d,"retry_used":%s,"fetch_latency_ms":%d}',
        query, len(results), image_count, dedup_removed,
        str(retry_used).lower(), fetch_latency_ms,
    )

    return {
        "results": results,
        "count": len(results),
        "query": query,
        "_meta": {
            "fetch_latency_ms": fetch_latency_ms,
            "article_count_raw": article_count_raw,
            "article_count_deduped": article_count_deduped,
            "dedup_ratio": dedup_ratio,
            "image_availability_rate": image_availability_rate,
            "retry_used": retry_used,
            "source_diversity": source_diversity,
            "region": region,
            "language": language,
        },
    }


# ── Fetch ─────────────────────────────────────────────────────────────────────

def _fetch_with_retry(query: str, region: str, period: str, language: str):
    """Fetch articles, retrying once with 2s backoff on failure."""
    from GoogleNews import GoogleNews

    retry_used = False
    last_error = None

    for attempt in range(2):
        if attempt > 0:
            retry_used = True
            time.sleep(2)

        try:
            gn = GoogleNews(lang=language, region=region, encode="utf-8")
            if period:
                gn.set_period(period)
            gn.clear()
            gn.get_news(query)
            raw = gn.result()
            gn.clear()
            if raw is not None:
                articles = [_normalize_article(r) for r in raw if r.get("title", "").strip()]
                return articles, retry_used, None
        except Exception as e:
            last_error = e
            logger.warning(
                '{"event":"fetch_attempt_failed","query":"%s","attempt":%d,"error":"%s"}',
                query, attempt + 1, str(e)[:120],
            )

    return [], retry_used, last_error


def _normalize_article(raw: dict) -> dict:
    """Normalize a raw GoogleNews result into a clean article dict."""
    import dateparser

    title = (raw.get("title") or "").strip()
    source = (raw.get("media") or "").strip()
    date_str = (raw.get("date") or "").strip()
    description = (raw.get("desc") or "").strip()
    url = (raw.get("link") or "").strip()
    image_url = (raw.get("img") or "").strip()

    # Normalize relative URLs from Google News aggregator
    if url and not url.startswith("http"):
        url = f"https://news.google.com{url}" if url.startswith("/") else f"https://{url}"

    # Remove source-name prefix from description (common artifact)
    if description and source and description.lower().startswith(source.lower()):
        description = description[len(source):].lstrip(" -:").strip()

    # Normalize date to ISO 8601 (best-effort; keep original string as fallback)
    published_at = ""
    if date_str:
        try:
            parsed = dateparser.parse(
                date_str,
                settings={"RETURN_AS_TIMEZONE_AWARE": False, "PREFER_DAY_OF_MONTH": "first"},
            )
            if parsed:
                published_at = parsed.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            pass

    return {
        "title": title,
        "source": source,
        "date": date_str,
        "published_at": published_at,
        "description": description,
        "url": url,
        "image_url": image_url,
        "also_reported_by": [],
    }


# ── Deduplication ─────────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Strip query params and fragments for URL comparison."""
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    except Exception:
        return url


def _dedup_by_url(articles: list) -> list:
    """Remove exact-same articles (same normalized URL from different aggregator paths)."""
    seen_urls = {}
    for article in articles:
        key = _normalize_url(article["url"])
        if not key or key in seen_urls:
            continue
        seen_urls[key] = article
    return list(seen_urls.values())


def _title_words(title: str) -> set:
    """Extract normalized significant words from a title."""
    words = re.sub(r"[^\w\s]", "", title.lower()).split()
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union)


def _dedup_by_title(articles: list) -> list:
    """
    Merge articles with >50% Jaccard title-word similarity.
    Keeps the best article (has image > longer description > earlier position).
    Attaches cross-referenced source names to the kept article.
    """
    used = set()
    deduplicated = []

    for i, article in enumerate(articles):
        if i in used:
            continue

        words_i = _title_words(article["title"])
        also_reported_by = list(article.get("also_reported_by") or [])

        for j in range(i + 1, len(articles)):
            if j in used:
                continue
            words_j = _title_words(articles[j]["title"])
            if _jaccard(words_i, words_j) > 0.5:
                used.add(j)
                other = articles[j]
                other_source = other.get("source", "")
                if other_source and other_source != article.get("source", ""):
                    also_reported_by.append(other_source)

                # Prefer the better article (image > longer description > earlier)
                current_score = (
                    bool(article.get("image_url")),
                    len(article.get("description", "")),
                )
                other_score = (
                    bool(other.get("image_url")),
                    len(other.get("description", "")),
                )
                if other_score > current_score:
                    # Swap: keep the other article, carry over cross-refs
                    article = {**other, "also_reported_by": []}
                    if article.get("source") != articles[i].get("source"):
                        also_reported_by.append(articles[i].get("source", ""))
                    words_i = _title_words(article["title"])

        article = {**article, "also_reported_by": also_reported_by}
        deduplicated.append(article)

    return deduplicated


# ── Reranking ─────────────────────────────────────────────────────────────────

def _rerank(articles: list) -> list:
    """
    Light importance reranking:
    - Primary: multi-source count (bigger stories float up)
    - Secondary: freshness (published_at ISO string, lexicographic comparison is safe)

    Freshness is intentionally a weak secondary signal to avoid overweighting trivial updates.
    """
    def _score(article):
        source_count = len(article.get("also_reported_by") or [])
        published_at = article.get("published_at") or ""
        return (source_count, published_at)

    return sorted(articles, key=_score, reverse=True)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _classify_error(e: Exception) -> str:
    """Classify a fetch exception for structured logging."""
    msg = str(e).lower()
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "connection" in msg or "network" in msg or "unreachable" in msg:
        return "network"
    return "blocked_or_changed_layout"
