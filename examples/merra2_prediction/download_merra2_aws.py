#!/usr/bin/env python3
"""
Download MERRA-2 files to ./merra2.

Config-driven (no long CLI). Edit CONFIG below.

This script supports two modes:
1) "earthaccess": uses NASA Earthdata Login via earthaccess, following the
   GESDISC tutorial pattern (search_data + download).
2) "s3": direct S3/HTTPS downloads using bucket/prefix/pattern or a manifest.
"""

from __future__ import annotations

import concurrent.futures as futures
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import boto3
import botocore
import requests


# -----------------------------
# CONFIG (edit these values)
# -----------------------------
CONFIG = {
    # Output directory
    "output_dir": "./merra2",

    # Select mode: "earthaccess" (download via earthaccess), "s3" (direct S3),
    # or "list_s3_urls" (query CMR and write S3 URLs).
    "mode": "earthaccess",

    # --- Mode: earthaccess ---
    # Uses Earthdata Login + search to get MERRA-2 granules, then downloads.
    # You must have Earthdata credentials set up (see earthaccess docs).
    "earthaccess": {
        "daac": "GESDISC",
        "doi": "",  # Example DOI (adjust for your collection)
        #"short_name": "M2I3NVCHM",  # MERRA-2 inst3_3d_chm_Nv collection short_name
        "short_name": "M2I3NVASM",   # MERRA-2 inst3_3d_asm_Nv collection short_name
        "temporal_start": "2020-01-01",
        "temporal_end": "2020-01-05",
        "max_results": None,  # set an int to cap results
        "require_us_west_2": False,  # Set True if you require in-region access
    },

    # Output file for list_s3_urls mode
    "list_output": "./merra2_s3_urls.txt",

    # --- Mode: s3 ---
    # Option A: inline URL list (s3:// or https://)
    "urls": [
        "s3://gesdisc-cumulus-prod-protected/MERRA2/M2I3NVCHM.5.12.4/2020/01/MERRA2_400.inst3_3d_chm_Nv.20200101.nc4",
        "s3://gesdisc-cumulus-prod-protected/MERRA2/M2I3NVCHM.5.12.4/2020/01/MERRA2_400.inst3_3d_chm_Nv.20200102.nc4",
        "s3://gesdisc-cumulus-prod-protected/MERRA2/M2I3NVCHM.5.12.4/2020/01/MERRA2_400.inst3_3d_chm_Nv.20200103.nc4",
        "s3://gesdisc-cumulus-prod-protected/MERRA2/M2I3NVCHM.5.12.4/2020/01/MERRA2_400.inst3_3d_chm_Nv.20200104.nc4",
        "s3://gesdisc-cumulus-prod-protected/MERRA2/M2I3NVCHM.5.12.4/2020/01/MERRA2_400.inst3_3d_chm_Nv.20200105.nc4",
    ],

    # Option B: manifest file with one URL per line (s3:// or https://)
    # If set (non-empty), manifest is used and the S3 pattern settings below are ignored.
    "manifest": "",

    # Option C: generate S3 URLs from date range + pattern
    "s3_bucket": "YOUR_BUCKET",
    "s3_prefix": "YOUR_PREFIX",  # e.g. "MERRA2/inst3_3d_chm_Nv"
    "pattern": "MERRA2_400.inst3_3d_chm_Nv.{YYYY}{MM}{DD}.nc4",
    "start_date": "2020-01-01",
    "end_date": "2020-01-31",

    # Auth + behavior
    "anonymous": False,  # True for public buckets
    "max_workers": 4,
    "dry_run": False,

    # Optional AWS profile name from ~/.aws/credentials
    "aws_profile": "default",
    "credentials_file": "./aws_credentials",


    # If True, use Earthdata Login to fetch temporary AWS credentials
    # for protected NASA S3 buckets (recommended for gesdisc-cumulus-prod-protected).
    "use_earthaccess_creds": True,
}


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def date_range(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def read_manifest(path: Path) -> List[str]:
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def build_s3_urls(bucket: str, prefix: str, pattern: str, start: str, end: str) -> List[str]:
    if not (pattern and start and end):
        raise ValueError("pattern, start_date, and end_date are required to build keys")
    start_dt = parse_date(start)
    end_dt = parse_date(end)
    urls = []
    for d in date_range(start_dt, end_dt):
        y = d.strftime("%Y")
        m = d.strftime("%m")
        dd = d.strftime("%d")
        fname = pattern.format(YYYY=y, MM=m, DD=dd)
        key = "/".join([p for p in [prefix.strip("/"), fname] if p])
        urls.append(f"s3://{bucket}/{key}")
    return urls


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_s3(s3_client, bucket: str, key: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, key, str(out_path))


def download_http(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    # s3://bucket/key
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key





def get_earthaccess_s3_client(cfg: dict):
    try:
        import earthaccess
    except Exception as e:
        raise ImportError("earthaccess is required for use_earthaccess_creds=True. Install it first.") from e

    # Login prompts for Earthdata credentials if not already configured.
    auth = earthaccess.login()

    daac = None
    if isinstance(cfg, dict):
        daac = cfg.get("earthaccess", {}).get("daac") or cfg.get("daac")

    creds = None

    # Try auth-bound credential helper if available
    try:
        if hasattr(auth, "get_s3_credentials"):
            creds = auth.get_s3_credentials(daac=daac) if daac else auth.get_s3_credentials()
    except Exception:
        creds = None

    # Fallback to module-level helper
    if creds is None:
        try:
            if daac:
                creds = earthaccess.get_s3_credentials(daac=daac)
            else:
                creds = earthaccess.get_s3_credentials()
        except Exception:
            creds = None

    # earthaccess may return different shapes depending on version.
    if isinstance(creds, dict) and "credentials" in creds:
        creds = creds["credentials"]
    if isinstance(creds, (list, tuple)) and creds:
        creds = creds[0]

    key_map = {
        "accessKeyId": "aws_access_key_id",
        "secretAccessKey": "aws_secret_access_key",
        "sessionToken": "aws_session_token",
        "access_key": "aws_access_key_id",
        "secret_key": "aws_secret_access_key",
        "token": "aws_session_token",
        "aws_access_key_id": "aws_access_key_id",
        "aws_secret_access_key": "aws_secret_access_key",
        "aws_session_token": "aws_session_token",
    }

    session_kwargs = {}
    if isinstance(creds, dict):
        for k, v in key_map.items():
            if k in creds:
                session_kwargs[v] = creds[k]

    region = None
    if isinstance(creds, dict):
        region = creds.get("region") or creds.get("regionName")

    if session_kwargs:
        session = boto3.session.Session(
            region_name=region or "us-west-2",
            **session_kwargs,
        )
        return session.client("s3")

    # Fallback: try earthaccess AWSSession if available
    try:
        if hasattr(earthaccess, "aws") and hasattr(earthaccess.aws, "AWSSession"):
            aws_sess = earthaccess.aws.AWSSession(daac=daac) if daac else earthaccess.aws.AWSSession()
            if hasattr(aws_sess, "get_session"):
                session = aws_sess.get_session()
            else:
                session = aws_sess.session
            return session.client("s3")
    except Exception:
        pass

    keys = list(creds.keys()) if isinstance(creds, dict) else [type(creds).__name__]
    raise KeyError(
        "Earthaccess did not return expected S3 credential keys. "
        f"Got keys: {keys}. If this persists, set AWS credentials via "
        "CONFIG['credentials_file'] or CONFIG['aws_profile'], or use mode='earthaccess' "
        "to download via HTTPS instead of direct S3."
    )





def get_boto3_session(cfg: dict) -> boto3.session.Session:
    # Optionally point boto3 to a local credentials file.
    creds_path = cfg.get("credentials_file")
    if creds_path:
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = creds_path

    profile = cfg.get("aws_profile")
    try:
        if profile:
            return boto3.session.Session(profile_name=profile)
        return boto3.session.Session()
    except botocore.exceptions.ProfileNotFound:
        # Fallback to default profile if the requested one is missing.
        return boto3.session.Session()


def check_region(require_us_west_2: bool) -> None:
    region = boto3.client("s3").meta.region_name
    if require_us_west_2 and region != "us-west-2":
        raise ValueError("This script is not running in us-west-2; direct S3 access may fail.")
    if region != "us-west-2":
        print(f"Warning: AWS region is {region}; NASA S3 access is typically in us-west-2.")


def run_earthaccess(cfg: dict, out_dir: Path) -> int:
    try:
        import earthaccess
    except Exception as e:
        raise ImportError("earthaccess is required for mode='earthaccess'. Install it first.") from e

    check_region(cfg.get("require_us_west_2", False))

    earthaccess.login()
    kwargs = {
        "temporal": (cfg["temporal_start"], cfg["temporal_end"]),
    }
    if cfg.get("doi"):
        kwargs["doi"] = cfg["doi"]
    if cfg.get("short_name"):
        kwargs["short_name"] = cfg["short_name"]

    results = earthaccess.search_data(**kwargs)
    if cfg.get("max_results"):
        results = results[: cfg["max_results"]]

    if not results:
        print("No results found for the query.")
        return 2

    if CONFIG.get("dry_run"):
        for r in results:
            print(r)
        return 0

    # earthaccess.download API changed across versions; try both signatures.
    try:
        earthaccess.download(results, path=str(out_dir))
    except TypeError:
        earthaccess.download(results, local_path=str(out_dir))
    return 0

    earthaccess.download(results, path=str(out_dir))
    return 0





def list_s3_urls(cfg: dict) -> list:
    try:
        import earthaccess
    except Exception as e:
        raise ImportError("earthaccess is required for list_s3_urls. Install it first.") from e

    earthaccess.login()
    kwargs = {
        "temporal": (cfg["temporal_start"], cfg["temporal_end"]),
    }
    if cfg.get("doi"):
        kwargs["doi"] = cfg["doi"]
    if cfg.get("short_name"):
        kwargs["short_name"] = cfg["short_name"]

    results = earthaccess.search_data(**kwargs)
    if cfg.get("max_results"):
        results = results[: cfg["max_results"]]

    urls = []
    for g in results:
        for link in g.data_links():
            if isinstance(link, str) and link.startswith("s3://"):
                urls.append(link)
    return urls


def run_s3(cfg: dict, out_dir: Path) -> int:
    urls: List[str] = []
    if cfg.get("manifest"):
        urls = read_manifest(Path(cfg["manifest"]))
    elif cfg.get("urls"):
        urls = list(cfg["urls"])
    else:
        if not cfg.get("s3_bucket"):
            raise ValueError("CONFIG['s3_bucket'] must be set")
        urls = build_s3_urls(cfg["s3_bucket"], cfg["s3_prefix"], cfg["pattern"], cfg["start_date"], cfg["end_date"])

    if not urls:
        print("No URLs to download")
        return 2

    if cfg.get("dry_run"):
        for u in urls:
            print(u)
        return 0

    if cfg.get("use_earthaccess_creds"):
        s3_client = get_earthaccess_s3_client(cfg)
    else:
        session = get_boto3_session(cfg)
        if cfg.get("anonymous"):
            s3_client = session.client("s3", config=botocore.client.Config(signature_version=botocore.UNSIGNED))
        else:
            s3_client = session.client("s3")

    def job(url: str) -> str:
        if url.startswith("s3://"):
            bucket, key = parse_s3_uri(url)
            out_path = out_dir / Path(key).name
            download_s3(s3_client, bucket, key, out_path)
            return str(out_path)
        if url.startswith("http://") or url.startswith("https://"):
            out_path = out_dir / Path(url).name
            download_http(url, out_path)
            return str(out_path)
        raise ValueError(f"Unsupported URL: {url}")

    errors = 0
    with futures.ThreadPoolExecutor(max_workers=cfg["max_workers"]) as ex:
        futs = {ex.submit(job, u): u for u in urls}
        for f in futures.as_completed(futs):
            u = futs[f]
            try:
                path = f.result()
                print(f"Downloaded: {u} -> {path}")
            except Exception as e:
                errors += 1
                print(f"Failed: {u} ({e})")

    return 1 if errors else 0


def main() -> int:
    out_dir = Path(CONFIG["output_dir"])
    ensure_output_dir(out_dir)

    mode = CONFIG.get("mode", "s3")
    if mode == "earthaccess":
        return run_earthaccess(CONFIG["earthaccess"], out_dir)
    if mode == "list_s3_urls":
        urls = list_s3_urls(CONFIG["earthaccess"])
        if not urls:
            print("No S3 URLs found.")
            return 2
        list_path = Path(CONFIG["list_output"])
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(urls) + "\n")
        print(f"Wrote {len(urls)} S3 URLs to {list_path}")
        return 0
    if mode == "s3":
        return run_s3(CONFIG, out_dir)
    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
