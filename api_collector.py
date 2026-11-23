"""
api_collector.py
================
Collects HOURLY historical weather (temperature + relative humidity) from the
Open-Meteo Historical Weather API for the five Makkah locations and writes one
combined CSV.

    Source : https://open-meteo.com/en/docs/historical-weather-api
    Endpoint: https://archive-api.open-meteo.com/v1/archive
    Period : 2026-05-25 .. 2026-05-30 (Hajj season, as required by the project)

Output CSV columns (exactly as required):
    location, timestamp, temperature_2m, relative_humidity_2m

Besides the CSV it writes `collection_metadata.json` (when/how the data was
collected + data-quality report), so nothing about the dataset is undocumented.

Coordinates are NOT hard-coded here: they live in `locations.json`, each with
its source. A location without verified coordinates is never invented -- the
collector stops and tells you which one is missing (or skips it if you pass
--allow-partial).

Usage:
    python api_collector.py
    python api_collector.py --allow-partial      # collect only locations that have coordinates
    python api_collector.py --out historical_weather.csv --start 2026-05-25 --end 2026-05-30
"""

import argparse
import itertools
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2026-05-25"
END_DATE = "2026-05-30"
HOURLY_VARS = ["temperature_2m", "relative_humidity_2m"]
TIMEZONE = "Asia/Riyadh"  # Makkah local time (UTC+3); otherwise the API returns GMT
OUT_COLUMNS = ["location", "timestamp", "temperature_2m", "relative_humidity_2m"]

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger("api_collector")


class CollectorError(Exception):
    """Raised for any problem that should stop the collection with a clear message."""


# --------------------------------------------------------------------------- #
# 1. Locations
# --------------------------------------------------------------------------- #
def load_locations(path):
    """Read locations.json and split into (usable, missing_coordinates)."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    usable, missing = [], []
    for item in items:
        lat, lon = item.get("latitude"), item.get("longitude")
        if lat is None or lon is None:
            missing.append(item["name"])
        elif not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise CollectorError(f"Invalid coordinates for {item['name']}: {lat}, {lon}")
        else:
            usable.append(item)
    return usable, missing


# --------------------------------------------------------------------------- #
# 2. API call with error handling
# --------------------------------------------------------------------------- #
def fetch_location(session, name, lat, lon, start, end, retries=4, timeout=30):
    """
    Download hourly data for one location and return a DataFrame with OUT_COLUMNS.

    Retries (exponential backoff 2s, 4s, 8s...) on: timeouts, connection errors,
    HTTP 429 (rate limit) and 5xx. Other HTTP errors (e.g. 400 bad parameters)
    are not retried because retrying cannot fix them.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(API_URL, params=params, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code} (retryable)")
            if resp.status_code != 200:
                # Open-Meteo sends {"error": true, "reason": "..."} for bad requests.
                try:
                    reason = resp.json().get("reason", resp.text[:200])
                except ValueError:
                    reason = resp.text[:200]
                raise CollectorError(f"{name}: HTTP {resp.status_code} - {reason}")
            return _parse_response(name, resp.json())
        except CollectorError:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            wait = 2 ** attempt
            log.warning("%s: attempt %d/%d failed (%s)", name, attempt, retries, exc)
            if attempt < retries:
                log.info("%s: waiting %ds before retry", name, wait)
                time.sleep(wait)
    raise CollectorError(f"{name}: failed after {retries} attempts. Last error: {last_error}")


def _parse_response(name, payload):
    """Turn the Open-Meteo JSON into a DataFrame, validating its structure."""
    hourly = payload.get("hourly")
    if not hourly:
        raise CollectorError(f"{name}: response has no 'hourly' section: {str(payload)[:200]}")
    for key in ["time"] + HOURLY_VARS:
        if key not in hourly:
            raise CollectorError(f"{name}: response is missing '{key}'")
    lengths = {len(hourly[k]) for k in ["time"] + HOURLY_VARS}
    if len(lengths) != 1:
        raise CollectorError(f"{name}: hourly arrays have different lengths {lengths}")
    df = pd.DataFrame(
        {
            "location": name,
            "timestamp": hourly["time"],
            "temperature_2m": hourly["temperature_2m"],
            "relative_humidity_2m": hourly["relative_humidity_2m"],
        }
    )
    return df[OUT_COLUMNS]


# --------------------------------------------------------------------------- #
# 3. Data quality
# --------------------------------------------------------------------------- #
def quality_report(df, start, end):
    """
    Check the combined dataframe. Nothing is dropped silently: duplicates are
    removed by clean(), missing values are KEPT (as empty cells) and counted here.
    """
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    expected = days * 24
    report = {
        "expected_rows_per_location": expected,
        "rows_total": int(len(df)),
        "rows_per_location": {k: int(v) for k, v in df.groupby("location").size().items()},
        "duplicate_rows_found": int(df.duplicated(subset=["location", "timestamp"]).sum()),
        "missing_values": {
            c: int(df[c].isna().sum()) for c in ["temperature_2m", "relative_humidity_2m"]
        },
        "out_of_range": {
            "relative_humidity_outside_0_100": int(
                ((df["relative_humidity_2m"] < 0) | (df["relative_humidity_2m"] > 100)).sum()
            ),
            "temperature_outside_minus10_60C": int(
                ((df["temperature_2m"] < -10) | (df["temperature_2m"] > 60)).sum()
            ),
        },
        "warnings": [],
    }
    for loc, n in report["rows_per_location"].items():
        if n != expected:
            report["warnings"].append(f"{loc}: {n} rows, expected {expected}")

    # Open-Meteo is gridded (reanalysis), so two close neighbourhoods can fall in the
    # same grid cell and return identical numbers. Flag that instead of hiding it.
    series = {
        loc: tuple(zip(g["temperature_2m"], g["relative_humidity_2m"]))
        for loc, g in df.groupby("location")
    }
    for a, b in itertools.combinations(series, 2):
        if series[a] == series[b]:
            report["warnings"].append(
                f"{a} and {b} have IDENTICAL values: probably the same Open-Meteo grid cell "
                f"(the API is gridded, not station data)."
            )
    return report


def clean(df):
    """Remove exact duplicate (location, timestamp) rows; keep the first."""
    return df.drop_duplicates(subset=["location", "timestamp"], keep="first").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def collect(locations_path, out_csv, meta_path, start, end, allow_partial):
    usable, missing = load_locations(locations_path)
    if missing and not allow_partial:
        raise CollectorError(
            "No verified coordinates for: " + ", ".join(missing) + ". Fill them in "
            f"{locations_path} (with a source), or run with --allow-partial to collect only the others."
        )
    if not usable:
        raise CollectorError("No location with coordinates to collect.")
    if missing:
        log.warning("Skipping locations without coordinates: %s", ", ".join(missing))

    frames = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "makkah-hajj-student-project/1.0"})
        for loc in usable:
            log.info("Fetching %s (%.5f, %.5f)", loc["name"], loc["latitude"], loc["longitude"])
            frames.append(fetch_location(session, loc["name"], loc["latitude"], loc["longitude"], start, end))
            time.sleep(0.5)  # be polite to the free API

    raw = pd.concat(frames, ignore_index=True)
    report = quality_report(raw, start, end)  # measured BEFORE cleaning
    df = clean(raw)
    df.to_csv(out_csv, index=False)
    log.info("Wrote %d rows to %s", len(df), out_csv)

    metadata = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Open-Meteo Historical Weather API",
        "endpoint": API_URL,
        "docs": "https://open-meteo.com/en/docs/historical-weather-api",
        "period": {"start": start, "end": end},
        "variables": HOURLY_VARS,
        "timezone_param": TIMEZONE,
        "weather_model": "API default ('best match'); not overridden",
        "locations_collected": [
            {k: v for k, v in l.items() if k in ("name", "latitude", "longitude", "source")} for l in usable
        ],
        "locations_skipped_no_coordinates": missing,
        "quality_report": report,
        "rows_in_csv": int(len(df)),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info("Wrote metadata/quality report to %s", meta_path)
    for w in report["warnings"]:
        log.warning("QUALITY WARNING: %s", w)
    return df, metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--locations", default=str(BASE_DIR / "locations.json"))
    p.add_argument("--out", default=str(BASE_DIR / "historical_weather.csv"))
    p.add_argument("--meta", default=str(BASE_DIR / "collection_metadata.json"))
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=END_DATE)
    p.add_argument("--allow-partial", action="store_true", help="skip locations that have no coordinates")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        collect(args.locations, args.out, args.meta, args.start, args.end, args.allow_partial)
    except CollectorError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
