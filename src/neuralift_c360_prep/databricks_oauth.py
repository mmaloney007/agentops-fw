#!/usr/bin/env python3
"""
Headless Databricks OAuth helpers.

Purpose:
    - Exchange client_id/client_secret for an access token without browser login.
    - Cache tokens in-process to avoid repeated requests.

Author: Mike Maloney - Neuralift, Inc.
Updated: 2025-01-07
Copyright © 2026 Neuralift, Inc.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_TOKEN_CACHE: Dict[Tuple[str, str, str], Tuple[str, float]] = {}


def _normalize_host(host: str) -> str:
    host = (host or "").strip().rstrip("/")
    if not host:
        return ""
    if not host.startswith(("http://", "https://")):
        host = f"https://{host}"
    return host


def _fetch_json(url: str) -> dict:
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=30) as resp:
            payload = resp.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Databricks OIDC fetch failed: {exc.code} {exc.reason} {body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Databricks OIDC fetch failed: {exc.reason}") from exc
    return json.loads(payload)


def _post_form(url: str, data: dict) -> dict:
    body = urlencode(data).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=30) as resp:
            payload = resp.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Databricks token request failed: {exc.code} {exc.reason} {body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Databricks token request failed: {exc.reason}") from exc
    return json.loads(payload)


def get_databricks_access_token(
    *,
    host: str,
    client_id: str,
    client_secret: str,
    scope: str | None = None,
) -> str:
    """
    Fetch a workspace access token via client_credentials (headless OAuth).
    """
    override = os.getenv("DATABRICKS_OAUTH_ACCESS_TOKEN")
    if override:
        return override

    host_norm = _normalize_host(host)
    if not host_norm:
        raise ValueError("DATABRICKS_HOST is required to request an access token")

    scope = scope or os.getenv("DATABRICKS_OAUTH_SCOPE", "all-apis")
    cache_key = (host_norm, client_id, scope)
    cached = _TOKEN_CACHE.get(cache_key)
    if cached:
        token, exp = cached
        if time.time() < (exp - 60):
            return token

    oidc = _fetch_json(f"{host_norm}/oidc/.well-known/oauth-authorization-server")
    token_url = oidc.get("token_endpoint")
    if not token_url:
        raise RuntimeError("OIDC metadata missing token_endpoint")

    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    if scope:
        payload["scope"] = scope

    token_data = _post_form(token_url, payload)
    access_token = token_data.get("access_token")
    if not access_token:
        raise RuntimeError("Token response missing access_token")
    expires_in = int(token_data.get("expires_in", 3600))
    _TOKEN_CACHE[cache_key] = (access_token, time.time() + expires_in)
    return access_token


__all__ = ["get_databricks_access_token"]
