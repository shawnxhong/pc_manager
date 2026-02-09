# tools.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
from pathlib import Path
from typing import Optional

import requests
from langchain_core.tools import tool

from launcher import PCManager
from pc_manager_tables import MS_SETTINGS_ITEMS, CONTROL_PANEL_ITEMS, SYSTEM_TOOL_ITEMS

# Index cache directory
INDEX_DIR = Path(__file__).resolve().parent / ".pc_manager_index"
_AUDIO_EXECUTOR: ThreadPoolExecutor | None = None

# Global singleton with lock to avoid concurrent double-init
_PC: Optional[PCManager] = None
_PC_LOCK = threading.Lock()


def _get_pc() -> PCManager:
    global _PC
    if _PC is not None:
        return _PC
    with _PC_LOCK:
        if _PC is None:
            _PC = PCManager(
                ms_settings_items=MS_SETTINGS_ITEMS,
                control_panel_items=CONTROL_PANEL_ITEMS,
                system_tool_items=SYSTEM_TOOL_ITEMS,
                index_dir=INDEX_DIR,
                embedder_model="BAAI/bge-m3",
                embedder_device="cpu",
            )
    return _PC


def release_rag_resource() -> None:
    global _PC
    with _PC_LOCK:
        if _PC is not None:
            try:
                _PC.close()
            except Exception:
                pass
        _PC = None
    import gc

    gc.collect()


def _com_thread_init():
    # Called once when the audio COM thread starts
    from comtypes import CoInitializeEx, COINIT_MULTITHREADED

    try:
        CoInitializeEx(COINIT_MULTITHREADED)
    except OSError as e:
        # -2147417850 = RPC_E_CHANGED_MODE, already initialized in another mode
        if getattr(e, "winerror", None) != -2147417850:
            raise
    except Exception:
        from comtypes import CoInitialize

        CoInitialize()


def _get_audio_executor():
    global _AUDIO_EXECUTOR
    if _AUDIO_EXECUTOR is None:
        _AUDIO_EXECUTOR = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="AudioCOM",
            initializer=_com_thread_init,
        )
    return _AUDIO_EXECUTOR


def _safe_run(name, fn, timeout_s: int):
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn)
            return fut.result(timeout=timeout_s)
    except _Timeout:
        return json.dumps(
            {"ok": False, "tool": name, "error": f"timeout>{timeout_s}s"},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"ok": False, "tool": name, "error": f"{type(e).__name__}: {e}"},
            ensure_ascii=False,
        )


@tool
def pc_manager_search(intent: str, top_k: int = 5) -> str:
    """
    PC Manager RAG search: given a user intent, return top matching Windows tools
    from ms-settings / control panel / exe&msc catalog.
    The assistant MUST NOT invent any URI; it must pick from returned candidates.
    """
    intent = (intent or "").strip()
    if not intent:
        return json.dumps({"ok": False, "error": "missing intent"}, ensure_ascii=False)

    pc = _get_pc()
    r = pc.resolve(intent, top_k=top_k)
    return json.dumps(r.__dict__, ensure_ascii=False)


@tool
def pc_manager_open(
    intent: str,
    reason: Optional[str] = None,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    min_margin: Optional[float] = None,
    target_id: Optional[str] = None,
) -> str:
    """
    Open the best matching Windows tool for a given user intent. It uses a
    persisted RAG index over ms-settings / control panel / exe&msc catalog. If
    ambiguous, it returns candidates and does NOT open anything.
    """
    intent = (intent or "").strip()
    if not intent:
        return json.dumps({"ok": False, "error": "missing intent"}, ensure_ascii=False)

    pc = _get_pc()
    out = pc.open(
        intent=intent,
        reason=reason,
        top_k=top_k or 3,
        min_score=min_score or 0.35,
        min_margin=min_margin or 0.005,
        target_id=target_id,
    )
    return json.dumps(out, ensure_ascii=False)


@tool
def image_generation(prompt: str) -> str:
    """
    AI painting (image generation) service, input text description, and return the
    image URL drawn based on text information.
    """

    def _impl():
        encoded_prompt = urllib.parse.quote(prompt)
        return json.dumps(
            {"ok": True, "image_url": f"https://image.pollinations.ai/prompt/{encoded_prompt}"},
            ensure_ascii=False,
        )

    return _safe_run("image_generation", _impl, 20)


@tool
def wikipedia(query: str) -> str:
    """A wrapper around Wikipedia. Useful for general facts."""

    def _impl():
        from langchain.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper

        wikipedia_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
        )
        result = wikipedia_tool.run(query)
        return json.dumps({"ok": True, "result": str(result)}, ensure_ascii=False)

    return _safe_run("wikipedia", _impl, 20)


# ---- Weather helpers ----

def _detect_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _normalize_lang(value: str) -> str:
    value = (value or "").lower().strip()
    if value.startswith("zh"):
        return "zh"
    if value.startswith("en"):
        return "en"
    return value


def _weather_ok(payload):
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def _weather_err(msg):
    return json.dumps({"ok": False, "error": msg}, ensure_ascii=False)


def _fetch_weather_by_city(city: str, unit: str, lang: str) -> str:
    """Fetch weather from wttr.in using a city name."""
    try:
        enc_city = urllib.parse.quote(city)
        url = f"https://wttr.in/{enc_city}?format=j1"
        if lang:
            url = f"{url}&lang={urllib.parse.quote(lang)}"
        headers = {"User-Agent": "agentic-weather/1.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        cc = (data.get("current_condition") or [{}])[0]
        area = (data.get("nearest_area") or [{}])[0]

        if unit == "fahrenheit":
            temp = cc.get("temp_F")
            feels = cc.get("FeelsLikeF")
            temp_unit = "fahrenheit"
        else:
            temp = cc.get("temp_C")
            feels = cc.get("FeelsLikeC")
            temp_unit = "celsius"

        desc = ""
        wd = cc.get("weatherDesc")
        if isinstance(wd, list) and wd:
            desc = wd[0].get("value", "")

        payload = {
            "source": "wttr.in",
            "query": {"city": city, "unit": temp_unit, "lang": lang or None},
            "location": {
                "name": " ".join(
                    filter(
                        None,
                        [
                            (area.get("areaName") or [{}])[0].get("value", ""),
                            (area.get("region") or [{}])[0].get("value", ""),
                            (area.get("country") or [{}])[0].get("value", ""),
                        ],
                    )
                ).strip()
                or city,
                "latitude": float(
                    ((area.get("latitude") or "0").split(",")[0]).strip() or 0.0
                )
                if isinstance(area.get("latitude"), str)
                else None,
                "longitude": float(
                    ((area.get("longitude") or "0").split(",")[0]).strip() or 0.0
                )
                if isinstance(area.get("longitude"), str)
                else None,
            },
            "current": {
                "temperature": float(temp) if temp not in (None, "") else None,
                "feels_like": float(feels) if feels not in (None, "") else None,
                "humidity_percent": float(cc.get("humidity"))
                if cc.get("humidity") not in (None, "")
                else None,
                "wind_kph": float(cc.get("windspeedKmph"))
                if cc.get("windspeedKmph") not in (None, "")
                else None,
                "description": desc,
                "observation_time": cc.get("observation_time"),
                "unit": temp_unit,
            },
            "language": lang or None,
        }
        return _weather_ok(payload)
    except Exception as e:
        return _weather_err(f"wttr.in failed: {e}")


def _fetch_weather_by_coords(lat: float, lon: float, unit: str) -> str:
    """Fetch weather from Open-Meteo using latitude/longitude."""
    try:
        temp_unit = "fahrenheit" if unit == "fahrenheit" else "celsius"
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={float(lat)}&longitude={float(lon)}"
            f"&current_weather=true&temperature_unit={temp_unit}"
            "&windspeed_unit=kmh"
        )
        headers = {"User-Agent": "agentic-weather/1.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        cw = data.get("current_weather") or {}

        payload = {
            "source": "open-meteo",
            "query": {"lat": float(lat), "lon": float(lon), "unit": temp_unit},
            "location": {
                "name": None,
                "latitude": float(lat),
                "longitude": float(lon),
            },
            "current": {
                "temperature": cw.get("temperature"),
                "feels_like": None,
                "humidity_percent": None,
                "wind_kph": cw.get("windspeed"),
                "description": None,
                "observation_time": cw.get("time"),
                "unit": temp_unit,
            },
        }
        return _weather_ok(payload)
    except Exception as e:
        return _weather_err(f"open-meteo failed: {e}")


@tool
def realtime_weather(
    city: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    unit: str = "celsius",
    lang: Optional[str] = None,
) -> str:
    """
    Query real-time weather by city name or latitude/longitude.
    - If 'city' is provided: uses wttr.in (no API key).
    - If 'lat' and 'lon' are provided: uses Open-Meteo (no API key).
    Fields are normalized to a common schema.
    lang: optional response language hint (e.g. "en", "zh").
    """
    city = (city or "").strip()
    unit = (unit or "celsius").lower()
    lang = (lang or "").lower()

    if not lang and city and _detect_cjk(city):
        lang = "zh"
    lang = _normalize_lang(lang)

    if city:
        return _fetch_weather_by_city(city, unit, lang)

    if lat is not None and lon is not None:
        return _fetch_weather_by_coords(lat, lon, unit)

    return _weather_err("Please provide either 'city' or both 'lat' and 'lon'.")


def get_langgraph_tools():
    return [
        pc_manager_search,
        pc_manager_open,
        image_generation,
        wikipedia,
        realtime_weather,
    ]
