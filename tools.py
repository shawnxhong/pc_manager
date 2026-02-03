# tools.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
from pathlib import Path
from typing import Optional

import requests
from langchain_core.tools import tool

from launcher import PCManager
from pc_manager_tables import MS_SETTINGS_ITEMS, CONTROL_PANEL_ITEMS, SYSTEM_TOOL_ITEMS

# 索引落盘目录
INDEX_DIR = Path(__file__).resolve().parent / ".pc_manager_index"
_AUDIO_EXECUTOR: ThreadPoolExecutor | None = None

# 全局单例：避免每次 tool call 都重建索引/重载 embedder
_PC: Optional[PCManager] = None


def _get_pc() -> PCManager:
    global _PC
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
    if _PC is not None:
        try:
            _PC.close()
        except Exception:
            pass
    _PC = None
    import gc

    gc.collect()


def _com_thread_init():
    # 这个函数在音频线程启动时仅调用一次
    from comtypes import CoInitializeEx, COINIT_MULTITHREADED

    try:
        CoInitializeEx(COINIT_MULTITHREADED)
    except OSError as e:
        # -2147417850 = RPC_E_CHANGED_MODE, 已在别的模型初始化，继续用即可
        if getattr(e, "winerror", None) != -2147417850:
            raise
    except Exception:
        # 尝试最小化初始化
        from comtypes import CoInitialize

        CoInitialize()


def _get_audio_executor():
    global _AUDIO_EXECUTOR
    if _AUDIO_EXECUTOR is None:
        _AUDIO_EXECUTOR = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="AudioCOM",
            initializer=_com_thread_init,  # 在线程里初始化 COM，一直不 Uninitialize
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
def get_current_weather(city_name: str) -> str:
    """Get the current weather in a given city name."""

    def _impl():
        key_selection = {
            "current_condition": [
                "temp_C",
                "FeelsLikeC",
                "humidity",
                "weatherDesc",
                "observation_time",
            ]
        }
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        ret = {k: {_v: data[k][0][_v] for _v in v} for k, v in key_selection.items()}
        return json.dumps({"ok": True, "data": ret}, ensure_ascii=False)

    return _safe_run("get_current_weather", _impl, 20)


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


@tool
def set_system_volume(
    mode: str = "absolute",
    level: Optional[float] = None,
    mute: Optional[bool] = None,
) -> str:
    """
    Set Windows system master volume. Supports absolute/relative level, mute/unmute, and query.
    """

    def _impl():
        def _volume_impl(pdict):
            # 这一段代码在“AudioCOM”专用线程里运行
            from ctypes import POINTER, cast
            from comtypes import CLSCTX_ALL, GUID
            from comtypes.client import CreateObject

            # 兼容不同版本 pycaw 的常量位置
            from pycaw import pycaw as pycaw_mod

            AudioUtilities = pycaw_mod.AudioUtilities
            IAudioEndpointVolume = pycaw_mod.IAudioEndpointVolume
            IMMDeviceEnumerator = pycaw_mod.IMMDeviceEnumerator

            if hasattr(pycaw_mod, "EDataFlow") and hasattr(pycaw_mod, "ERole"):
                eRender = pycaw_mod.EDataFlow.eRender.value
                eMultimedia = pycaw_mod.ERole.eMultimedia.value
            else:
                from pycaw import constants as pycaw_constants

                eRender = pycaw_constants.eRender
                eMultimedia = pycaw_constants.eMultimedia

            def clamp01(x):
                try:
                    x = float(x)
                except Exception:
                    return 0.0
                if x != x:
                    return 0.0
                return max(0.0, min(1.0, x))

            mode_local = pdict.get("mode", "absolute")
            level_local = pdict.get("level", None)
            mute_local = pdict.get("mute", None)

            # 获取 endpoint（先走老接口，再走标准枚举器）
            try:
                dev = AudioUtilities.GetSpeakers()
                iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                endpoint = cast(iface, POINTER(IAudioEndpointVolume))
            except Exception:
                try:
                    enum = CreateObject("MMDeviceEnumerator", interface=IMMDeviceEnumerator)
                except Exception:
                    enum = CreateObject(
                        GUID("{BCDE0395-E52F-467C-8E3D-C4579291692E}"),
                        interface=IMMDeviceEnumerator,
                    )
                ep = enum.GetDefaultAudioEndpoint(eRender, eMultimedia)
                iface = ep.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                endpoint = cast(iface, POINTER(IAudioEndpointVolume))

            # 读取当前
            before_scalar = float(endpoint.GetMasterVolumeLevelScalar())
            before_percent = round(before_scalar * 100, 1)
            before_mute = bool(endpoint.GetMute())

            # 调音量模式且未显式传 mute -> 自动取消静音
            auto_unmute = (mode_local in ("absolute", "relative")) and (mute_local is None)
            if auto_unmute and before_mute:
                endpoint.SetMute(0, None)

            # 调整音量
            if mode_local == "absolute" and level_local is not None:
                endpoint.SetMasterVolumeLevelScalar(clamp01(level_local / 100.0), None)
            elif mode_local == "relative" and level_local is not None:
                endpoint.SetMasterVolumeLevelScalar(
                    clamp01(before_scalar + (level_local / 100.0)), None
                )
            elif mode_local not in ("query", "mute"):
                mode_local = "query"

            # 静音控制
            if mode_local == "mute":
                mute_local = True if mute_local is None else bool(mute_local)
                endpoint.SetMute(1 if mute_local else 0, None)
            elif mute_local is not None:
                endpoint.SetMute(1 if mute_local else 0, None)

            # 最终状态
            after_scalar = float(endpoint.GetMasterVolumeLevelScalar())
            after_percent = round(after_scalar * 100, 1)
            after_mute = bool(endpoint.GetMute())

            return {
                "ok": True,
                "before": {"volume_percent": before_percent, "mute": before_mute},
                "after": {"volume_percent": after_percent, "mute": after_mute},
                "mode": mode_local,
                "auto_unmute": auto_unmute,
            }

        # 通过常驻音频线程执行
        try:
            fut = _get_audio_executor().submit(
                _volume_impl, {"mode": mode, "level": level, "mute": mute}
            )
            res = fut.result(timeout=3.0)  # 可按需调大
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            return json.dumps(
                {"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False
            )

    return _safe_run("set_system_volume", _impl, 20)


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
    """

    city = (city or "").strip()
    unit = (unit or "celsius").lower()
    lang = (lang or "").lower()

    def _ok(payload):
        return json.dumps({"ok": True, **payload}, ensure_ascii=False)

    def _err(msg):
        return json.dumps({"ok": False, "error": msg}, ensure_ascii=False)

    # --- 使用城市名：wttr.in ---
    if city:
        try:
            enc_city = urllib.parse.quote(city)
            url = f"https://wttr.in/{enc_city}?format=j1"
            headers = {"User-Agent": "agentic-weather/1.0"}
            resp = requests.get(url, headers=headers, timeout=8)
            resp.raise_for_status()
            data = resp.json()

            cc = (data.get("current_condition") or [{}])[0]
            area = (data.get("nearest_area") or [{}])[0]
            # 按单位取温度/体感温度
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
            }
            return _ok(payload)
        except Exception as e:
            return _err(f"wttr.in failed: {e}")

    # --- 使用经纬度：Open-Meteo ---
    if lat is not None and lon is not None:
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
                    "feels_like": None,  # Open-Meteo current_weather 无体感温度
                    "humidity_percent": None,  # 需额外字段，这里保持简洁
                    "wind_kph": cw.get("windspeed"),
                    "description": None,  # 需要天气码翻译，可按需扩展
                    "observation_time": cw.get("time"),
                    "unit": temp_unit,
                },
            }
            return _ok(payload)
        except Exception as e:
            return _err(f"open-meteo failed: {e}")

    return _err("Please provide either 'city' or both 'lat' and 'lon'.")


def get_langgraph_tools():
    return [
        pc_manager_search,
        pc_manager_open,
        image_generation,
        get_current_weather,
        wikipedia,
        realtime_weather,
    ]
