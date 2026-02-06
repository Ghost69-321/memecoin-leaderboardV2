# streamlit_app.py
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import httpx
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ConfigDict

APP_NAME = "Memecoin Leaderboard"
BASE_URL_DEFAULT = "https://lunarcrush.com/api4"
API_PUBLIC_SEGMENT = "/public"  # can be toggled off if needed
API_ENDPOINT = "/coins/list/v1"

# ------------------------
# Config & Secrets
# ------------------------
def get_settings() -> Dict[str, Any]:
    token = st.secrets.get("LUNARCRUSH_TOKEN") or os.getenv("LUNARCRUSH_TOKEN")
    base_url = st.secrets.get("LUNARCRUSH_BASE_URL") or os.getenv("LUNARCRUSH_BASE_URL") or BASE_URL_DEFAULT
    timeout_s = int(st.secrets.get("HTTP_TIMEOUT_S", 30))
    cache_ttl_s = int(st.secrets.get("CACHE_TTL_S", 300))
    if not token:
        st.error("LUNARCRUSH_TOKEN is missing. Add it in Settings → Secrets (Streamlit Cloud) or .streamlit/secrets.toml locally.")
        st.stop()
    return {
        "token": token.strip(),
        "base_url": base_url.rstrip("/"),
        "timeout_s": timeout_s,
        "cache_ttl_s": cache_ttl_s,
    }

# ------------------------
# Types
# ------------------------
class Coin(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str
    name: Optional[str] = None
    galaxy_score: Optional[float] = None
    alt_rank: Optional[int] = Field(default=None, alias="altrank")
    price_usd: Optional[float] = None
    market_cap_usd: Optional[float] = None
    volume_24h: Optional[float] = None
    url_slug: Optional[str] = None

# ------------------------
# HTTP + Retry
# ------------------------
def backoff_sleep(attempt: int) -> float:
    import random
    base = min(2 ** attempt, 32)
    return base + random.uniform(0, 0.5)

def make_client(timeout_s: int) -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(timeout_s, read=timeout_s))

def build_headers(token: str, auth_mode: str) -> Dict[str, str]:
    if auth_mode == "Bearer":
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    elif auth_mode == "X-API-Key":
        return {"X-API-Key": token, "Accept": "application/json"}
    elif auth_mode == "LC-API-Key":
        return {"LC-API-Key": token, "Accept": "application/json"}
    else:
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

def fetch_memecoins(
    base_url: str,
    token: str,
    limit: int,
    page: int,
    sort_field: str,
    desc: bool,
    filter_value: str = "memecoins",
    max_attempts: int = 3,
    auth_mode: str = "Bearer",
    use_public_segment: bool = True,
) -> Dict[str, Any]:
    params = {"sort": sort_field, "filter": filter_value, "limit": limit, "page": page}
    if desc:
        params["desc"] = "true"

    base = base_url
    if use_public_segment:
        base += API_PUBLIC_SEGMENT
    url = f"{base}{API_ENDPOINT}"

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            with make_client(timeout_s=settings["timeout_s"]) as client:
                r = client.get(url, headers=build_headers(token, auth_mode), params=params)
                if r.status_code == 429:
                    time.sleep(backoff_sleep(attempt))
                    continue
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as he:
                    # sanitize token out of preview
                    preview = r.text[:300].replace(token, "***")
                    raise RuntimeError(f"HTTP {r.status_code} {url} → {preview}") from he
                return r.json()
        except httpx.HTTPError as e:
            last_err = e
            time.sleep(backoff_sleep(attempt))
    raise RuntimeError(f"Failed to fetch data after {max_attempts} attempts: {last_err}")

# ------------------------
# Caching layer
# ------------------------
@st.cache_data(show_spinner=False, ttl=300)
def cached_fetch(
    base_url: str,
    token: str,
    limit: int,
    page: int,
    sort_field: str,
    desc: bool,
    filter_value: str,
    auth_mode: str,
    use_public_segment: bool,
) -> Dict[str, Any]:
    return fetch_memecoins(base_url, token, limit, page, sort_field, desc, filter_value, auth_mode=auth_mode, use_public_segment=use_public_segment)

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🚀", layout="wide")
st.title(APP_NAME)
st.caption("Rank memecoin candidates using Galaxy Score™ / AltRank™ (LunarCrush v4).")

settings = get_settings()

with st.sidebar:
    st.header("Controls")
    sort_mode = st.selectbox("Sort by", options=["galaxy_score (desc)", "alt_rank (asc)"], index=0)
    if "galaxy" in sort_mode:
        sort_field = "galaxy_score"
        desc = True
    else:
        sort_field = "alt_rank"
        desc = False

    page_size = st.number_input("Page size (limit)", min_value=10, max_value=250, value=100, step=10)
    page = st.number_input("Page index", min_value=0, max_value=1000, value=0, step=1)
    top_k = st.number_input("Show top K", min_value=5, max_value=100, value=20, step=5)

    allow_list = st.text_input("Allow symbols (comma-separated)", value="")
    deny_list = st.text_input("Deny symbols (comma-separated)", value="")

    st.markdown("---")
    st.subheader("API Settings")
    auth_mode = st.selectbox("Auth header mode", options=["Bearer", "X-API-Key", "LC-API-Key"], index=0, help="Some accounts use API key headers instead of Bearer tokens.")
    use_public_segment = st.checkbox("Use /public segment", value=True, help="Try unchecking if you consistently see 401/404.")

    st.markdown("---")
    with st.expander("Advanced"):
        st.write(f"Base URL: `{settings['base_url']}` (toggle /public above)")
        st.write(f"Cache TTL (s): {settings['cache_ttl_s']}")
        st.write(f"HTTP timeout (s): {settings['timeout_s']}")
        masked = settings["token"][:4] + "…" + settings["token"][-4:]
        st.write(f"Token (masked): {masked} (length {len(settings['token'])})")

    st.markdown("---")
    st.caption("Tip: Add your LUNARCRUSH_TOKEN in **Settings → Secrets**.")
    if st.button("Test API now"):
        try:
            _raw = fetch_memecoins(
                settings["base_url"],
                settings["token"],
                1, 0, "galaxy_score", True, "memecoins",
                max_attempts=1, auth_mode=auth_mode, use_public_segment=use_public_segment
            )
            st.success("API OK ✓ — got data")
        except Exception as e:
            st.error(f"API test failed: {e}")

# Fetch
with st.spinner("Fetching leaderboard..."):
    raw = cached_fetch(
        settings["base_url"],
        settings["token"],
        int(page_size),
        int(page),
        sort_field,
        desc,
        "memecoins",
        auth_mode,
        use_public_segment,
    )

data = raw.get("data", []) if isinstance(raw, dict) else []

# Normalize to DataFrame
def normalize(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    records = []
    for r in rows:
        try:
            c = Coin.model_validate(r)
            records.append({
                "timestamp_utc": c.timestamp_utc.isoformat(),
                "symbol": c.symbol,
                "name": c.name,
                "galaxy_score": c.galaxy_score,
                "alt_rank": c.alt_rank,
                "price_usd": c.price_usd,
                "market_cap_usd": c.market_cap_usd,
                "volume_24h": c.volume_24h,
                "url_slug": c.url_slug,
            })
        except Exception:
            records.append({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": r.get("symbol"),
                "name": r.get("name"),
                "galaxy_score": r.get("galaxy_score"),
                "alt_rank": r.get("altrank") or r.get("alt_rank"),
                "price_usd": r.get("price_usd") or r.get("price"),
                "market_cap_usd": r.get("market_cap_usd") or r.get("market_cap"),
                "volume_24h": r.get("volume_24h") or r.get("volume"),
                "url_slug": r.get("url_slug") or r.get("id"),
            })
    return pd.DataFrame.from_records(records)

df = normalize(data)

# Apply allow/deny filters
def parse_csv_list(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]

allow = set(parse_csv_list(allow_list))
deny = set(parse_csv_list(deny_list))

if allow:
    df = df[df["symbol"].str.upper().isin(allow)]
if deny:
    df = df[~df["symbol"].str.upper().isin(deny)]

# Sort client-side to be safe
if sort_field == "galaxy_score":
    df = df.sort_values(by=["galaxy_score"], ascending=False, na_position="last")
else:
    df = df.sort_values(by=["alt_rank"], ascending=True, na_position="last")

# "New in top K" badge using session state
state_key = f"top_seen_{sort_field}"
prev_top = set(st.session_state.get(state_key, []))
current_top = list(df["symbol"].head(int(top_k)).astype(str))
st.session_state[state_key] = current_top
now_new = [s for s in current_top if s not in prev_top]

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Top results")
with col2:
    st.metric("Rows", len(df))

st.dataframe(df.head(int(top_k)), use_container_width=True, hide_index=True)

if now_new:
    st.success("New in top K this refresh: " + ", ".join(now_new))

with st.expander("Health & Debug"):
    st.write({
        "fetched_rows": len(df),
        "sort_field": sort_field,
        "desc": desc,
        "page": int(page),
        "limit": int(page_size),
        "refresh_utc": datetime.now(timezone.utc).isoformat(),
        "auth_mode": auth_mode,
        "use_public_segment": use_public_segment,
    })

st.caption("Source: LunarCrush API v4.")
