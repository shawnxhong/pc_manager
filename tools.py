# tools_all.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, urllib.parse, requests
from qwen_agent.tools.base import BaseTool, register_tool
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
from functools import partial
from pathlib import Path
from typing import Optional
import gc
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
    gc.collect()

def _com_thread_init():
    # 这个函数在音频线程启动时仅调用一次
    try:
        from comtypes import CoInitializeEx, COINIT_MULTITHREADED
        CoInitializeEx(COINIT_MULTITHREADED)
    except OSError as e:
        # -2147417850 = RPC_E_CHANGED_MODE, 已在别的模型初始化，继续用即可
        if getattr(e, "winerror", None) != -2147417850:
            raise
    except Exception:
        # 尝试最小化初始化
        try:
            from comtypes import CoInitialize
            CoInitialize()
        except Exception:
            pass

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
        return json.dumps({"ok": False, "tool": name, "error": f"timeout>{timeout_s}s"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "tool": name, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


def register_tools(tool_timeout_s: int = 20):
    """
    注册并启用所有工具；返回工具名列表。
    Qwen-Agent 会使用 function_list=返回的列表 来启用工具。
    """


    @register_tool("pc_manager_search")
    class PCManagerSearch(BaseTool):
        description = (
            "PC Manager RAG search: given a user intent, return top matching Windows tools "
            "from ms-settings / control panel / exe&msc catalog. "
            "The assistant MUST NOT invent any URI; it must pick from returned candidates."
        )
        parameters = [
            {"name": "intent", "type": "string", "description": "User intent in natural language", "required": True},
            {"name": "top_k", "type": "number", "description": "Number of candidates", "required": False},
        ]
    
        def call(self, params: str, **kwargs) -> str:
            try:
                p = json.loads(params) if isinstance(params, str) else (params or {})
                intent = (p.get("intent") or "").strip()
                top_k = int(p.get("top_k") or 5)
                if not intent:
                    return json.dumps({"ok": False, "error": "missing intent"}, ensure_ascii=False)
    
                pc = _get_pc()
                r = pc.resolve(intent, top_k=top_k)
                return json.dumps(r.__dict__, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
 
 
    @register_tool("pc_manager_open")
    class PCManagerOpen(BaseTool):
        description = (
            "Open the best matching Windows tool for a given user intent. "
            "It uses a persisted RAG index over ms-settings / control panel / exe&msc catalog. "
            "If ambiguous, it returns candidates and does NOT open anything."
        )
        parameters = [
            {"name": "intent", "type": "string", "description": "User intent in natural language", "required": True},
            {"name": "reason", "type": "string", "description": "Optional explanation", "required": False},
            {"name": "top_k", "type": "number", "description": "Candidates to consider", "required": False},
            {"name": "min_score", "type": "number", "description": "Auto-open threshold", "required": False},
            {"name": "min_margin", "type": "number", "description": "Ambiguity margin between top1 and top2", "required": False},
            {"name": "target_id", "type": "string", "description": "If set, open this exact tool_id", "required": False},
        ]
    
        def call(self, params: str, **kwargs) -> str:
            try:
                p = json.loads(params) if isinstance(params, str) else (params or {})
                intent = (p.get("intent") or "").strip()
                reason = p.get("reason")
                # top_k = int(p.get("top_k") or 5)
                # min_score = float(p.get("min_score") or 0.34)
                # min_margin = float(p.get("min_margin") or 0.03)
                target_id = p.get("target_id")
                top_k = 3
                min_score = 0.35
                min_margin = 0.005
    
                if not intent:
                    return json.dumps({"ok": False, "error": "missing intent"}, ensure_ascii=False)
    
                pc = _get_pc()
                out = pc.open(
                    intent=intent,
                    reason=reason,
                    top_k=top_k,
                    min_score=min_score,
                    min_margin=min_margin,
                    target_id=target_id,
                )
                return json.dumps(out, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


    @register_tool("image_generation")
    class ImageGeneration(BaseTool):
        description = "AI painting (image generation) service, input text description, and return the image URL drawn based on text information."
        parameters = [{
            "name": "prompt", "type": "string",
            "description": "Detailed description of the desired image content, in English",
            "required": True
        }]
        # def call(self, params: str, **kwargs) -> str:
        #     prompt = json.loads(params)["prompt"]
        #     prompt = urllib.parse.quote(prompt)
        #     return json.dumps({"image_url": f"https://image.pollinations.ai/prompt/{prompt}"}, ensure_ascii=False)
        def call(self, params: str, **kwargs) -> str:
            def _impl():
                p = json.loads(params) if isinstance(params, str) else params
                prompt = urllib.parse.quote(p["prompt"])
                return json.dumps({"ok": True, "image_url": f"https://image.pollinations.ai/prompt/{prompt}"}, ensure_ascii=False)
            return _safe_run("image_generation", _impl, tool_timeout_s)

    @register_tool("get_current_weather")
    class GetCurrentWeather(BaseTool):
        description = "Get the current weather in a given city name."
        parameters = [{
            "name": "city_name", "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
            "required": True
        }]
        # def call(self, params: str, **kwargs) -> str:
        #     city_name = json.loads(params)["city_name"]
        #     key_selection = {"current_condition": ["temp_C","FeelsLikeC","humidity","weatherDesc","observation_time"]}
        #     resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
        #     resp.raise_for_status()
        #     data = resp.json()
        #     ret = {k: {v: data[k][0][v] for v in vs} for k, vs in key_selection.items()}
        #     return str(ret)
        def call(self, params: str, **kwargs) -> str:
            def _impl():
                p = json.loads(params) if isinstance(params, str) else params
                city = p["city_name"]
                key_selection = {"current_condition": ["temp_C","FeelsLikeC","humidity","weatherDesc","observation_time"]}
                resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
                resp.raise_for_status()
                data = resp.json()
                ret = {k: {_v: data[k][0][_v] for _v in v} for k, v in key_selection.items()}
                return json.dumps({"ok": True, "data": ret}, ensure_ascii=False)
            return _safe_run("get_current_weather", _impl, tool_timeout_s)

    @register_tool("wikipedia")
    class Wikipedia(BaseTool):
        description = "A wrapper around Wikipedia. Useful for general facts."
        parameters = [{
            "name": "query", "type": "string",
            "description": "Query to look up on wikipedia",
            "required": True
        }]
        # def call(self, params: str, **kwargs) -> str:
        #     from langchain.tools import WikipediaQueryRun
        #     from langchain_community.utilities import WikipediaAPIWrapper
        #     query = json.loads(params)["query"]
        #     wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000))
        #     return str(wikipedia.run(query))
        def call(self, params: str, **kwargs) -> str:
            def _impl():
                p = json.loads(params) if isinstance(params, str) else params
                from langchain.tools import WikipediaQueryRun
                from langchain_community.utilities import WikipediaAPIWrapper
                wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000))
                result = wikipedia.run(p["query"])
                return json.dumps({"ok": True, "result": str(result)}, ensure_ascii=False)
            return _safe_run("wikipedia", _impl, tool_timeout_s)

    @register_tool("set_system_volume")
    class SetSystemVolume(BaseTool):
        """
        Set Windows system master volume. Supports absolute/relative level, mute/unmute, and query.
        """
        description = "Set Windows system master volume. Supports absolute/relative level, mute/unmute, and query."
        parameters = [
            {"name":"mode","type":"string","description":"How to change volume: 'absolute', 'relative', 'query'.",
             "enum":["absolute","relative","query"],"required":False},
            {"name":"level","type":"number","description":"Percent. absolute:0..100; relative:-100..100.","required":False},
            {"name":"mute","type":"boolean","description":"Set mute state (optional).","required":False},
        ]        
        def call(self, params: str, **kwargs) -> str:
            def _impl():
                p = json.loads(params) if isinstance(params, str) else params
                
                def _volume_impl(pdict):
                    # 这一段代码在“AudioCOM”专用线程里运行
                    from ctypes import POINTER, cast
                    from comtypes import CLSCTX_ALL, GUID
                    from comtypes.client import CreateObject
                    # 兼容不同版本 pycaw 的常量位置
                    try:
                        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, IMMDeviceEnumerator, EDataFlow, ERole
                        eRender = EDataFlow.eRender.value
                        eMultimedia = ERole.eMultimedia.value
                    except Exception:
                        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, IMMDeviceEnumerator
                        from pycaw.constants import eRender, eMultimedia

                    def clamp01(x):
                        try:
                            x = float(x)
                        except Exception:
                            return 0.0
                        if x != x:
                            return 0.0
                        return max(0.0, min(1.0, x))

                    mode  = pdict.get("mode", "absolute")
                    level = pdict.get("level", None)
                    mute  = pdict.get("mute",  None)

                    # 获取 endpoint（先走老接口，再走标准枚举器）
                    try:
                        dev = AudioUtilities.GetSpeakers()
                        iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                        endpoint = cast(iface, POINTER(IAudioEndpointVolume))
                    except Exception:
                        try:
                            enum = CreateObject("MMDeviceEnumerator", interface=IMMDeviceEnumerator)
                        except Exception:
                            enum = CreateObject(GUID('{BCDE0395-E52F-467C-8E3D-C4579291692E}'), interface=IMMDeviceEnumerator)
                        ep = enum.GetDefaultAudioEndpoint(eRender, eMultimedia)
                        iface = ep.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                        endpoint = cast(iface, POINTER(IAudioEndpointVolume))

                    # 读取当前
                    before_scalar = float(endpoint.GetMasterVolumeLevelScalar())
                    before_percent = round(before_scalar * 100, 1)
                    before_mute = bool(endpoint.GetMute())

                    # 调音量模式且未显式传 mute -> 自动取消静音
                    auto_unmute = (mode in ("absolute", "relative")) and (mute is None)
                    if auto_unmute and before_mute:
                        endpoint.SetMute(0, None)

                    # 调整音量
                    if mode == "absolute" and level is not None:
                        endpoint.SetMasterVolumeLevelScalar(clamp01(level / 100.0), None)
                    elif mode == "relative" and level is not None:
                        endpoint.SetMasterVolumeLevelScalar(clamp01(before_scalar + (level / 100.0)), None)
                    elif mode not in ("query", "mute"):
                        mode = "query"

                    # 静音控制
                    if mode == "mute":
                        mute = True if mute is None else bool(mute)
                        endpoint.SetMute(1 if mute else 0, None)
                    elif mute is not None:
                        endpoint.SetMute(1 if mute else 0, None)

                    # 最终状态
                    after_scalar = float(endpoint.GetMasterVolumeLevelScalar())
                    after_percent = round(after_scalar * 100, 1)
                    after_mute = bool(endpoint.GetMute())

                    return {
                        "ok": True,
                        "before": {"volume_percent": before_percent, "mute": before_mute},
                        "after":  {"volume_percent": after_percent,  "mute": after_mute},
                        "mode": mode,
                        "auto_unmute": auto_unmute
                    }

                # 通过常驻音频线程执行
                try:
                    fut = _get_audio_executor().submit(_volume_impl, p)
                    res = fut.result(timeout=3.0)  # 可按需调大
                    return json.dumps(res, ensure_ascii=False)
                except Exception as e:
                    return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
    
            return _safe_run("set_system_volume", _impl, tool_timeout_s)

        
    @register_tool("realtime_weather")
    class RealtimeWeather(BaseTool):
        """
        Query real-time weather by city name or latitude/longitude.
        - If 'city' is provided: uses wttr.in (no API key).
        - If 'lat' and 'lon' are provided: uses Open-Meteo (no API key).
        Fields are normalized to a common schema.
        """
        description = "Get current weather by city name or (lat, lon). No API key required."
        parameters = [
            {
                "name": "city",
                "type": "string",
                "description": "City name, e.g. 'San Francisco, CA' or '北京'. If provided, lat/lon are ignored.",
                "required": False
            },
            {
                "name": "lat",
                "type": "number",
                "description": "Latitude in decimal degrees (used when city is not provided).",
                "required": False
            },
            {
                "name": "lon",
                "type": "number",
                "description": "Longitude in decimal degrees (used when city is not provided).",
                "required": False
            },
            {
                "name": "unit",
                "type": "string",
                "description": "Temperature unit.",
                "enum": ["celsius", "fahrenheit"],
                "required": False
            },
            {
                "name": "lang",
                "type": "string",
                "description": "Language hint (e.g. 'en', 'zh'). Only applied to wttr.in.",
                "required": False
            }
        ]

        def call(self, params: str, **kwargs) -> str:
            import requests, urllib.parse

            p = json.loads(params) if isinstance(params, str) else params
            city = (p.get("city") or "").strip()
            lat  = p.get("lat", None)
            lon  = p.get("lon", None)
            unit = (p.get("unit") or "celsius").lower()
            lang = (p.get("lang") or "").lower()

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
                            "name": " ".join(filter(None, [
                                (area.get("areaName") or [{}])[0].get("value", ""),
                                (area.get("region") or [{}])[0].get("value", ""),
                                (area.get("country") or [{}])[0].get("value", "")
                            ])).strip() or city,
                            "latitude":  float(((area.get("latitude")  or "0").split(",")[0]).strip() or 0.0) if isinstance(area.get("latitude"), str) else None,
                            "longitude": float(((area.get("longitude") or "0").split(",")[0]).strip() or 0.0) if isinstance(area.get("longitude"), str) else None,
                        },
                        "current": {
                            "temperature": float(temp) if temp not in (None, "") else None,
                            "feels_like": float(feels) if feels not in (None, "") else None,
                            "humidity_percent": float(cc.get("humidity")) if cc.get("humidity") not in (None, "") else None,
                            "wind_kph": float(cc.get("windspeedKmph")) if cc.get("windspeedKmph") not in (None, "") else None,
                            "description": desc,
                            "observation_time": cc.get("observation_time"),
                            "unit": temp_unit
                        }
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
                            "unit": temp_unit
                        }
                    }
                    return _ok(payload)
                except Exception as e:
                    return _err(f"open-meteo failed: {e}")

            return _err("Please provide either 'city' or both 'lat' and 'lon'.")
    
    @register_tool("system_hardware_info")
    class SystemHardwareInfo(BaseTool):
        """
        Query Windows PC hardware info: SoC/CPU, SKU, memory size & speed, disk sizes, battery level & health.
        Uses WMI (via pywin32) and psutil. No internet needed.

        This tool returns a markdown report with tables. It supports Chinese ('zh') and English ('en')
        via the optional 'lang' parameter. After calling this tool, the assistant SHOULD output the
        markdown directly, WITHOUT wrapping it in code fences.
        """
        description = (
            "Query Windows PC hardware info and return a markdown table. "
            "After calling this tool, the assistant MUST:\n"
            "1. Output the markdown table EXACTLY as provided by the tool.\n"
            "2. MUST NOT wrap it inside code fences (no triple backticks).\n"
            "3. MUST NOT convert it to plain text or summarize.\n"
            "4. MUST render markdown tables directly.\n"
            "5. Do NOT add any extra summary or bullet points after the tables."
        )
        parameters = [
            {
                "name": "lang",
                "type": "string",
                "description": "Language of the markdown report: 'zh' for Chinese, 'en' for English. "
                               "If omitted, the assistant should choose based on user language.",
                "required": True
            }
        ]

        def call(self, params: str = "", **kwargs) -> str:
            import json
            import psutil

            # ---------- 解析 lang 参数 ----------
            lang = "auto"
            if params:
                try:
                    p = json.loads(params) if isinstance(params, str) else params
                    lang = (p.get("lang") or "auto").lower()
                except Exception:
                    lang = "auto"
            # 这里不做复杂语言检测，默认英文；让 LLM 显式传 'zh' 就是中文表格
            if lang not in ("zh", "en"):
                lang = "en"

            # ---------- 小工具函数 ----------
            def _safe_int(x):
                try:
                    return int(str(x).strip())
                except Exception:
                    return None

            def _fmt_bytes_gib(x):
                if not x:
                    return ""
                try:
                    return f"{x / (1024 ** 3):.1f} GiB"
                except Exception:
                    return str(x)

            def _wmi_query(namespace, query, attrs):
                """
                安全 WMI 查询：
                - 在函数内部 CoInitialize（如果需要）
                - 只返回 Python dict 列表，不把 COM 对象往外传
                """
                try:
                    import pythoncom
                    import win32com.client
                except Exception as e:
                    raise RuntimeError(f"pywin32 not available: {e}") from e

                pythoncom.CoInitialize()  # 不 CoUninitialize，避免 GC 阶段释放崩溃
                results = []
                try:
                    locator = win32com.client.Dispatch("WbemScripting.SWbemLocator")
                    svc = locator.ConnectServer(".", namespace)
                    rows = svc.ExecQuery(query)
                    for r in rows:
                        item = {}
                        for a in attrs:
                            try:
                                val = getattr(r, a, None)
                            except Exception:
                                val = None
                            item[a] = val
                        results.append(item)
                except Exception:
                    raise
                finally:
                    # 不再 CoUninitialize，本线程保持 COM 初始化到进程结束
                    pass
                return results

            # ---------- CPU / SoC / SKU / Model ----------
            cpu_name = None
            cpu_vendor = None
            system_sku = None
            system_model = None
            cpu_base_mhz = None      # 基础频率（MHz）
            cpu_cur_mhz = None 
            
            try:
                rows = _wmi_query(
                    r"root\CIMV2",
                    "SELECT Name, Manufacturer, MaxClockSpeed, CurrentClockSpeed FROM Win32_Processor",
                    ["Name", "Manufacturer", "MaxClockSpeed", "CurrentClockSpeed"],
                )
                if rows:
                    r0 = rows[0]
                    cpu_name = str(r0.get("Name") or "").strip() or None
                    cpu_vendor = str(r0.get("Manufacturer") or "").strip() or None

                    try:
                        base = r0.get("MaxClockSpeed", None)
                        cpu_base_mhz = int(base) if base is not None else None
                    except Exception:
                        cpu_base_mhz = None

                    try:
                        cur = r0.get("CurrentClockSpeed", None)
                        cpu_cur_mhz = int(cur) if cur is not None else None
                    except Exception:
                        cpu_cur_mhz = None
            except Exception:
                pass

            try:
                rows = _wmi_query(
                    r"root\CIMV2",
                    "SELECT SystemSKUNumber, Model FROM Win32_ComputerSystem",
                    ["SystemSKUNumber", "Model"],
                )
                if rows:
                    system_sku = (rows[0].get("SystemSKUNumber") or "").strip() or None
                    system_model = (rows[0].get("Model") or "").strip() or None
            except Exception:
                pass

            # 额外补充：有些厂把产品信息放在 Win32_ComputerSystemProduct
            if not system_model:
                try:
                    rows = _wmi_query(
                        r"root\CIMV2",
                        "SELECT Name FROM Win32_ComputerSystemProduct",
                        ["Name"],
                    )
                    if rows and rows[0].get("Name"):
                        system_model = str(rows[0]["Name"]).strip()
                except Exception:
                    pass

            # ---------- Memory（逐条 & 总量 & 频率） ----------
            mem_modules = []
            total_mem_bytes = 0
            mem_freqs = []

            try:
                rows = _wmi_query(
                    r"root\CIMV2",
                    "SELECT Capacity, Speed, ConfiguredClockSpeed, Manufacturer, PartNumber FROM Win32_PhysicalMemory",
                    ["Capacity", "Speed", "ConfiguredClockSpeed", "Manufacturer", "PartNumber"],
                )
                for r in rows:
                    cap = _safe_int(r.get("Capacity"))
                    spd = _safe_int(r.get("Speed"))
                    cfg = _safe_int(r.get("ConfiguredClockSpeed"))
                    man = (r.get("Manufacturer") or "").strip() or None
                    pn = (r.get("PartNumber") or "").strip() or None
                    if cap:
                        total_mem_bytes += cap
                    freq = cfg or spd  # 优先用配置频率
                    if freq:
                        mem_freqs.append(freq)
                    mem_modules.append({
                        "size_bytes": cap,
                        "speed_mhz": freq,
                        "manufacturer": man,
                        "part_number": pn,
                    })
            except Exception:
                pass

            # ---------- Disks（物理盘） ----------
            disks = []
            total_disk_bytes = 0

            try:
                rows = _wmi_query(
                    r"root\CIMV2",
                    "SELECT Model, Size, InterfaceType, MediaType, SerialNumber FROM Win32_DiskDrive",
                    ["Model", "Size", "InterfaceType", "MediaType", "SerialNumber"],
                )
                for r in rows:
                    size = _safe_int(r.get("Size"))
                    model = (r.get("Model") or "").strip() or None
                    iface = (r.get("InterfaceType") or "").strip() or None
                    media = (r.get("MediaType") or "").strip() or None
                    sn = (r.get("SerialNumber") or "").strip() or None
                    if size:
                        total_disk_bytes += size
                    disks.append({
                        "model": model,
                        "size_bytes": size,
                        "interface": iface,
                        "media_type": media,
                        "serial": sn,
                    })
            except Exception:
                pass

            # ---------- 分区信息（逻辑卷 & 已用/剩余空间） ----------
            partitions = []
            try:
                for p in psutil.disk_partitions(all=False):
                    try:
                        usage = psutil.disk_usage(p.mountpoint)
                    except Exception:
                        usage = None
                    partitions.append({
                        "device": p.device,
                        "mountpoint": p.mountpoint,
                        "fstype": p.fstype,
                        "total_bytes": getattr(usage, "total", None) if usage else None,
                        "used_bytes": getattr(usage, "used", None) if usage else None,
                        "free_bytes": getattr(usage, "free", None) if usage else None,
                        "percent_used": getattr(usage, "percent", None) if usage else None,
                    })
            except Exception:
                pass

            # ---------- Battery（电量&健康&续航时间） ----------
            battery_percent = None
            on_ac = None
            secs_left = None
            time_remain_str = None  # 格式化后的“x小时y分钟”

            try:
                b = psutil.sensors_battery()
                if b is not None:
                    battery_percent = float(b.percent) if b.percent is not None else None
                    on_ac = bool(b.power_plugged)
                    secs_left = getattr(b, "secsleft", None)

                    # 只在有意义的值时做格式化（>0）
                    if isinstance(secs_left, (int, float)) and secs_left > 0:
                        h = int(secs_left // 3600)
                        m = int((secs_left % 3600) // 60)
                        if lang == "zh":
                            if h == 0 and m == 0:
                                time_remain_str = "小于 1 分钟"
                            elif h == 0:
                                time_remain_str = f"{m} 分钟"
                            elif m == 0:
                                time_remain_str = f"{h} 小时"
                            else:
                                time_remain_str = f"{h} 小时 {m} 分钟"
                        else:
                            if h == 0 and m == 0:
                                time_remain_str = "less than 1 min"
                            elif h == 0:
                                time_remain_str = f"{m} min"
                            elif m == 0:
                                time_remain_str = f"{h} h"
                            else:
                                time_remain_str = f"{h} h {m} min"
            except Exception:
                pass

            design_capacity = None    # 设计容量（mWh）
            full_charge_cap = None    # 满充容量（mWh）
            health_percent = None

            try:
                rows = _wmi_query(
                    r"root\WMI",
                    "SELECT DesignedCapacity FROM BatteryStaticData",
                    ["DesignedCapacity", "designedCapacity"],
                )
                if rows:
                    row = rows[0]
                    dc = row.get("designedCapacity", None)
                    if dc is None:
                        dc = row.get("DesignedCapacity", None)
                    if dc is not None:
                        design_capacity = int(dc)

                rows = _wmi_query(
                    r"root\WMI",
                    "SELECT FullChargedCapacity FROM BatteryFullChargedCapacity",
                    ["FullChargedCapacity"],
                )
                if rows and rows[0].get("FullChargedCapacity") is not None:
                    full_charge_cap = int(rows[0]["FullChargedCapacity"])

                if design_capacity and full_charge_cap and design_capacity > 0:
                    health_percent = round(full_charge_cap / design_capacity * 100.0, 2)
            except Exception:
                pass

            # ---------- 原始 payload ----------
            soc = {
                "cpu_name": cpu_name,
                "cpu_vendor": cpu_vendor,
                "cpu_base_mhz": cpu_base_mhz,
                "cpu_cur_mhz": cpu_cur_mhz,
                "system_sku": system_sku,
                "system_model": system_model,
            }
            mem = {
                "total_bytes": total_mem_bytes or None,
                "modules": mem_modules,
                "suggested_speed_mhz": max(set(mem_freqs), key=mem_freqs.count) if mem_freqs else None,
            }
            storage = {
                "total_bytes": total_disk_bytes or None,
                "drives": disks,
                "partitions": partitions,
            }
            battery = {
                "percent": battery_percent,
                "ac_connected": on_ac,
                "health_percent": health_percent,
                "full_charge_capacity_mWh": full_charge_cap,
                "design_capacity_mWh": design_capacity,
                "time_remaining_sec": secs_left,
                "time_remaining_str": time_remain_str,
            }
            payload = {
                "ok": True,
                "soc": soc,
                "memory": mem,
                "storage": storage,
                "battery": battery,
            }

            # ---------- 多语言 label 定义 ----------
            if lang == "en":
                title_soc = "### 🧠 SoC / Basic Info"
                title_mem = "### 🧬 Memory"
                title_sto = "### 💾 Storage"
                title_part = "#### 🧱 Partitions"
                title_bat = "### 🔋 Battery"

                soc_table_header = ["Item", "Info"]
                soc_labels = {
                    "cpu": "CPU",
                    "vendor": "Vendor",
                    "model": "Model",
                    "sku": "System SKU",
                    "base_freq": "Base frequency (MHz)",
                    "cur_freq": "Current clock (MHz)",
                }

                mem_table_header = ["Slot", "Capacity", "Frequency (MHz)", "Vendor", "Part Number"]
                mem_total_prefix = "Total capacity"

                disk_table_header = ["Disk", "Model", "Capacity", "Interface", "Media Type", "Serial"]
                disk_total_prefix = "Total capacity"

                part_table_header = ["Partition", "Mount", "File system", "Total", "Used", "Free", "Usage%"]

                batt_table_header = ["Item", "Value"]
                batt_labels = {
                    "percent": "Battery level",
                    "ac": "On AC power",
                    "design": "Design capacity",
                    "full": "Full charge capacity",
                    "health": "Health",
                    "time": "Estimated time left",
                    "yes": "Yes",
                    "no": "No",
                }
            else:  # zh
                title_soc = "### 🧠 SoC / 基本信息"
                title_mem = "### 🧬 内存信息"
                title_sto = "### 💾 存储信息"
                title_part = "#### 🧱 分区信息"
                title_bat = "### 🔋 电池信息"

                soc_table_header = ["项目", "信息"]
                soc_labels = {
                    "cpu": "CPU",
                    "vendor": "厂商",
                    "model": "机型 / Model",
                    "sku": "系统 SKU",
                    "base_freq": "基础频率 (MHz)",
                    "cur_freq": "当前频率 (MHz)",
                }

                mem_table_header = ["槽位", "容量", "频率(MHz)", "厂商", "型号"]
                mem_total_prefix = "总容量"

                disk_table_header = ["磁盘", "型号", "容量", "接口", "媒介类型", "序列号"]
                disk_total_prefix = "总容量"

                part_table_header = ["分区", "挂载点", "文件系统", "总容量", "已用", "剩余", "使用率"]

                batt_table_header = ["项目", "数值"]
                batt_labels = {
                    "percent": "当前电量",
                    "ac": "是否接电源",
                    "design": "设计容量",
                    "full": "满充容量",
                    "health": "健康度",
                    "time": "预计剩余时间",
                    "yes": "是",
                    "no": "否",
                }

            # ---------- SoC Markdown ----------
            soc_md = [
                f"| {soc_table_header[0]} | {soc_table_header[1]} |",
                f"|{'-'*6}|{'-'*6}|",
                f"| {soc_labels['cpu']} | {cpu_name or ''} |",
                f"| {soc_labels['vendor']} | {cpu_vendor or ''} |",
                f"| {soc_labels['model']} | {system_model or ''} |",
                f"| {soc_labels['sku']} | {system_sku or ''} |",
                f"| {soc_labels['base_freq']} | {cpu_base_mhz if cpu_base_mhz is not None else ''} |",
                f"| {soc_labels['cur_freq']} | {cpu_cur_mhz if cpu_cur_mhz is not None else ''} |",
            ]
            soc_md = "\n".join(soc_md)

            # ---------- 内存 Markdown ----------
            mem_rows = [
                "| " + " | ".join(mem_table_header) + " |",
                "|" + "|".join(["------"] * len(mem_table_header)) + "|",
            ]
            for idx, m in enumerate(mem_modules):
                mem_rows.append(
                    f"| #{idx} | {_fmt_bytes_gib(m.get('size_bytes'))} | "
                    f"{m.get('speed_mhz') or ''} | "
                    f"{m.get('manufacturer') or ''} | "
                    f"{m.get('part_number') or ''} |"
                )
            if not mem_modules:
                mem_rows.append("| (none) | | | | |" if lang == "en" else "| (无) | | | | |")
            mem_md = "\n".join(mem_rows)
            total_mem_str = _fmt_bytes_gib(total_mem_bytes) if total_mem_bytes else ""
            mem_intro = (
                f"- {mem_total_prefix}: **{total_mem_str or ('unknown' if lang == 'en' else '未知')}**"
            )

            # ---------- 物理磁盘 Markdown ----------
            disk_rows = [
                "| " + " | ".join(disk_table_header) + " |",
                "|" + "|".join(["------"] * len(disk_table_header)) + "|",
            ]
            for idx, d in enumerate(disks):
                disk_rows.append(
                    f"| #{idx} | {d.get('model') or ''} | {_fmt_bytes_gib(d.get('size_bytes'))} | "
                    f"{d.get('interface') or ''} | {d.get('media_type') or ''} | {d.get('serial') or ''} |"
                )
            if not disks:
                disk_rows.append("| (none) | | | | | |" if lang == "en" else "| (无) | | | | | |")
            disk_md = "\n".join(disk_rows)
            total_disk_str = _fmt_bytes_gib(total_disk_bytes) if total_disk_bytes else ""
            disk_intro = (
                f"- {disk_total_prefix}: **{total_disk_str or ('unknown' if lang == 'en' else '未知')}**"
            )

            # ---------- 分区 Markdown ----------
            part_rows = [
                "| " + " | ".join(part_table_header) + " |",
                "|" + "|".join(["------"] * len(part_table_header)) + "|",
            ]
            for idx, p in enumerate(partitions):
                part_rows.append(
                    f"| #{idx} | {p.get('mountpoint') or ''} | {p.get('fstype') or ''} | "
                    f"{_fmt_bytes_gib(p.get('total_bytes'))} | "
                    f"{_fmt_bytes_gib(p.get('used_bytes'))} | "
                    f"{_fmt_bytes_gib(p.get('free_bytes'))} | "
                    f"{(str(p.get('percent_used')) + '%') if p.get('percent_used') is not None else ''} |"
                )
            if not partitions:
                part_rows.append("| (none) | | | | | | |" if lang == "en" else "| (无) | | | | | | |")
            part_md = "\n".join(part_rows)

            # ---------- 电池 Markdown ----------
            batt_rows = [
                "| " + " | ".join(batt_table_header) + " |",
                "|" + "|".join(["------"] * len(batt_table_header)) + "|",
            ]
            if on_ac is not None:
                ac_text = batt_labels["yes"] if on_ac else batt_labels["no"]
            else:
                ac_text = ""

            batt_rows.append(
                f"| {batt_labels['percent']} | {battery_percent if battery_percent is not None else ''}% |"
            )
            batt_rows.append(
                f"| {batt_labels['ac']} | {ac_text} |"
            )
            batt_rows.append(
                f"| {batt_labels['design']} | {design_capacity if design_capacity is not None else ''} mWh |"
            )
            batt_rows.append(
                f"| {batt_labels['full']} | {full_charge_cap if full_charge_cap is not None else ''} mWh |"
            )
            batt_rows.append(
                f"| {batt_labels['health']} | {health_percent if health_percent is not None else ''}% |"
            )
            batt_rows.append(
                f"| {batt_labels['time']} | {time_remain_str or ''} |"
            )
            batt_md = "\n".join(batt_rows)

            # ---------- 拼接最终 markdown ----------
            markdown_summary = (
                f"{title_soc}\n\n"
                + soc_md
                + f"\n\n{title_mem}\n\n"
                + mem_intro
                + "\n\n"
                + mem_md
                + f"\n\n{title_sto}\n\n"
                + disk_intro
                + "\n\n"
                + disk_md
                + f"\n\n{title_part}\n\n"
                + part_md
                + f"\n\n{title_bat}\n\n"
                + batt_md
            )

            return markdown_summary

    
    return [
            "wikipedia",
            # "realtime_weather", 
            "pc_manager_open",
            "pc_manager_search",
            # "image_generation",
            # "set_system_volume",
            #  "system_hardware_info", 
            ]
