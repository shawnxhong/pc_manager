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


@tool
def system_hardware_info(lang: str) -> str:
    """
    Query Windows PC hardware info: SoC/CPU, SKU, memory size & speed, disk sizes,
    battery level & health. Uses WMI (via pywin32) and psutil. No internet needed.

    This tool returns a markdown report with tables. It supports Chinese ('zh') and
    English ('en') via the optional 'lang' parameter. After calling this tool, the
    assistant SHOULD output the markdown directly, WITHOUT wrapping it in code fences.
    """
    import psutil
    import pythoncom
    import win32com.client

    # ---------- 解析 lang 参数 ----------
    lang = (lang or "auto").lower()
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
        finally:
            # 不再 CoUninitialize，本线程保持 COM 初始化到进程结束
            pass
        return results

    # ---------- CPU / SoC / SKU / Model ----------
    cpu_name = None
    cpu_vendor = None
    system_sku = None
    system_model = None
    cpu_base_mhz = None  # 基础频率（MHz）
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
            mem_modules.append(
                {
                    "size_bytes": cap,
                    "speed_mhz": freq,
                    "manufacturer": man,
                    "part_number": pn,
                }
            )
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
            disks.append(
                {
                    "model": model,
                    "size_bytes": size,
                    "interface": iface,
                    "media_type": media,
                    "serial": sn,
                }
            )
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
            partitions.append(
                {
                    "device": p.device,
                    "mountpoint": p.mountpoint,
                    "fstype": p.fstype,
                    "total_bytes": getattr(usage, "total", None) if usage else None,
                    "used_bytes": getattr(usage, "used", None) if usage else None,
                    "free_bytes": getattr(usage, "free", None) if usage else None,
                    "percent_used": getattr(usage, "percent", None) if usage else None,
                }
            )
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

    design_capacity = None  # 设计容量（mWh）
    full_charge_capacity = None  # 充满容量（mWh）

    try:
        rows = _wmi_query(
            r"root\WMI",
            "SELECT DesignCapacity, FullChargeCapacity FROM BatteryFullChargedCapacity",
            ["DesignCapacity", "FullChargeCapacity"],
        )
        if rows:
            design_capacity = _safe_int(rows[0].get("DesignCapacity"))
            full_charge_capacity = _safe_int(rows[0].get("FullChargeCapacity"))
    except Exception:
        pass

    battery_health = None
    if design_capacity and full_charge_capacity:
        battery_health = round(full_charge_capacity / design_capacity * 100, 1)

    # ---------- 输出 Markdown ----------
    lines = []

    if lang == "zh":
        lines.append("## 设备硬件信息")
        lines.append("")
        lines.append("### 处理器 / 机型")
        lines.append("")
        lines.append("| 项目 | 值 |")
        lines.append("| --- | --- |")
        lines.append(f"| CPU | {cpu_name or ''} |")
        lines.append(f"| 厂商 | {cpu_vendor or ''} |")
        lines.append(f"| 基础频率 | {cpu_base_mhz or ''} MHz |")
        lines.append(f"| 当前频率 | {cpu_cur_mhz or ''} MHz |")
        lines.append(f"| 机型 | {system_model or ''} |")
        lines.append(f"| SKU | {system_sku or ''} |")

        lines.append("")
        lines.append("### 内存")
        lines.append("")
        lines.append("| 模块 | 容量 | 频率 | 厂商 | 型号 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for i, m in enumerate(mem_modules, 1):
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    i,
                    _fmt_bytes_gib(m.get("size_bytes")),
                    f"{m.get('speed_mhz')} MHz" if m.get("speed_mhz") else "",
                    m.get("manufacturer") or "",
                    m.get("part_number") or "",
                )
            )
        lines.append(f"| **总计** | **{_fmt_bytes_gib(total_mem_bytes)}** |  |  |  |")

        lines.append("")
        lines.append("### 硬盘")
        lines.append("")
        lines.append("| 磁盘 | 容量 | 接口 | 类型 | 序列号 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for i, d in enumerate(disks, 1):
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    i,
                    _fmt_bytes_gib(d.get("size_bytes")),
                    d.get("interface") or "",
                    d.get("media_type") or "",
                    d.get("serial") or "",
                )
            )
        lines.append(f"| **总计** | **{_fmt_bytes_gib(total_disk_bytes)}** |  |  |  |")

        lines.append("")
        lines.append("### 分区")
        lines.append("")
        lines.append("| 盘符 | 文件系统 | 总容量 | 已用 | 可用 | 使用率 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for p in partitions:
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    p.get("mountpoint") or p.get("device") or "",
                    p.get("fstype") or "",
                    _fmt_bytes_gib(p.get("total_bytes")),
                    _fmt_bytes_gib(p.get("used_bytes")),
                    _fmt_bytes_gib(p.get("free_bytes")),
                    f"{p.get('percent_used')}%" if p.get("percent_used") is not None else "",
                )
            )

        lines.append("")
        lines.append("### 电池")
        lines.append("")
        lines.append("| 项目 | 值 |")
        lines.append("| --- | --- |")
        lines.append(
            f"| 电量 | {f'{battery_percent:.1f}%' if battery_percent is not None else ''} |"
        )
        lines.append(f"| 供电 | {'外接电源' if on_ac else '电池' if on_ac is not None else ''} |")
        lines.append(f"| 续航 | {time_remain_str or ''} |")
        lines.append(
            f"| 设计容量 | {f'{design_capacity} mWh' if design_capacity else ''} |"
        )
        lines.append(
            f"| 满充容量 | {f'{full_charge_capacity} mWh' if full_charge_capacity else ''} |"
        )
        lines.append(
            f"| 健康度 | {f'{battery_health:.1f}%' if battery_health is not None else ''} |"
        )
    else:
        lines.append("## Device Hardware Info")
        lines.append("")
        lines.append("### CPU / Model")
        lines.append("")
        lines.append("| Item | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| CPU | {cpu_name or ''} |")
        lines.append(f"| Vendor | {cpu_vendor or ''} |")
        lines.append(f"| Base Clock | {cpu_base_mhz or ''} MHz |")
        lines.append(f"| Current Clock | {cpu_cur_mhz or ''} MHz |")
        lines.append(f"| Model | {system_model or ''} |")
        lines.append(f"| SKU | {system_sku or ''} |")

        lines.append("")
        lines.append("### Memory")
        lines.append("")
        lines.append("| Module | Size | Speed | Vendor | Part Number |")
        lines.append("| --- | --- | --- | --- | --- |")
        for i, m in enumerate(mem_modules, 1):
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    i,
                    _fmt_bytes_gib(m.get("size_bytes")),
                    f"{m.get('speed_mhz')} MHz" if m.get("speed_mhz") else "",
                    m.get("manufacturer") or "",
                    m.get("part_number") or "",
                )
            )
        lines.append(f"| **Total** | **{_fmt_bytes_gib(total_mem_bytes)}** |  |  |  |")

        lines.append("")
        lines.append("### Storage")
        lines.append("")
        lines.append("| Disk | Size | Interface | Media | Serial |")
        lines.append("| --- | --- | --- | --- | --- |")
        for i, d in enumerate(disks, 1):
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    i,
                    _fmt_bytes_gib(d.get("size_bytes")),
                    d.get("interface") or "",
                    d.get("media_type") or "",
                    d.get("serial") or "",
                )
            )
        lines.append(f"| **Total** | **{_fmt_bytes_gib(total_disk_bytes)}** |  |  |  |")

        lines.append("")
        lines.append("### Partitions")
        lines.append("")
        lines.append("| Mount | FS | Total | Used | Free | Usage |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for p in partitions:
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    p.get("mountpoint") or p.get("device") or "",
                    p.get("fstype") or "",
                    _fmt_bytes_gib(p.get("total_bytes")),
                    _fmt_bytes_gib(p.get("used_bytes")),
                    _fmt_bytes_gib(p.get("free_bytes")),
                    f"{p.get('percent_used')}%" if p.get("percent_used") is not None else "",
                )
            )

        lines.append("")
        lines.append("### Battery")
        lines.append("")
        lines.append("| Item | Value |")
        lines.append("| --- | --- |")
        lines.append(
            f"| Charge | {f'{battery_percent:.1f}%' if battery_percent is not None else ''} |"
        )
        lines.append(
            f"| Power | {'AC' if on_ac else 'Battery' if on_ac is not None else ''} |"
        )
        lines.append(f"| Remaining | {time_remain_str or ''} |")
        lines.append(
            f"| Design Capacity | {f'{design_capacity} mWh' if design_capacity else ''} |"
        )
        lines.append(
            f"| Full Charge | {f'{full_charge_capacity} mWh' if full_charge_capacity else ''} |"
        )
        lines.append(
            f"| Health | {f'{battery_health:.1f}%' if battery_health is not None else ''} |"
        )

    return "\n".join(lines).strip()


def get_langgraph_tools():
    return [
        pc_manager_search,
        pc_manager_open,
        image_generation,
        get_current_weather,
        wikipedia,
        set_system_volume,
        realtime_weather,
        system_hardware_info,
    ]
