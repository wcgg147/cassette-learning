# -*- coding: utf-8 -*-
"""天气数据工具（2026-10-22，plans/1022-01 点3）。

为什么不用 web_search：Bing RSS 返回的是天气网站的**介绍页**，没有温度；
模型拿不到数值，只能编或让用户自己查。本工具直接调 Open-Meteo（免费、无 key）：
  地理编码：https://geocoding-api.open-meteo.com/v1/search
  预报：    https://api.open-meteo.com/v1/forecast

设计口径
--------
- **有城市才查**；没城市返回 `needs_location=True`，由回复侧自然追问
  （"你在哪个城市"），不猜、不默认北京。
- 只取当前 + 今天 + 明天，够回答"今天/明天天气"；不拉 15 天。
- 失败返回 `ok=False`，不抛异常；回复侧据此说"我这边查不到"。

用法（只读，不写库）：
    tool = WeatherTool()
    result = await tool.lookup("北京")
"""

from __future__ import annotations

import re
from typing import Any

import httpx

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# 常见城市：中文直接匹配；英文按词边界匹配。没列到的城市先返回 needs_location，
# 后续可以接 geocoding 的模糊搜索，不在这里堆词表。
_CITIES_ZH = (
    "北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "武汉", "西安",
    "南京", "天津", "苏州", "长沙", "郑州", "青岛", "大连", "厦门", "福州",
    "昆明", "贵阳", "南宁", "海口", "三亚", "哈尔滨", "长春", "沈阳",
    "石家庄", "太原", "济南", "合肥", "南昌", "兰州", "西宁", "银川",
    "乌鲁木齐", "呼和浩特", "拉萨", "香港", "澳门", "台北",
)
_CITIES_EN = (
    "Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou", "Chengdu",
    "Chongqing", "Wuhan", "Xi'an", "Xian", "Nanjing", "Tianjin", "Suzhou",
    "Changsha", "Qingdao", "Dalian", "Xiamen", "Kunming", "Harbin",
    "Shenyang", "Singapore", "Tokyo", "Seoul", "New York", "Los Angeles",
    "London", "Paris", "Sydney", "Vancouver", "Toronto",
)

_WEATHER_CODES = {
    0: "晴", 1: "大部晴朗", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def extract_city(text: str) -> str | None:
    """从用户话里取城市；取不到返回 None（由回复侧追问，不默认）。"""
    raw = str(text or "")
    for city in _CITIES_ZH:
        if city in raw:
            return city
    lowered = raw.casefold()
    for city in _CITIES_EN:
        if re.search(rf"\b{re.escape(city.casefold())}\b", lowered):
            return city
    return None


def wants_weather(text: str) -> bool:
    """只认带问句标记的天气问题；陈述句（今天天气不错）不触发查询。"""
    return bool(re.search(
        r"(天气|气温|温度|下雨|下雪|晴天|阴天|weather|forecast|temperature|rain|snow)"
        r".{0,12}(怎么样|如何|多少度|几度|吗|会不会|要下|有雨|有雪|"
        r"today|tomorrow|this\s+week|\?)",
        str(text or ""), re.IGNORECASE,
    ))


def describe_code(code: Any) -> str:
    try:
        return _WEATHER_CODES.get(int(code), "未知天气")
    except Exception:
        return "未知天气"


class WeatherTool:
    def __init__(self, *, timeout: float = 8.0) -> None:
        self.timeout = max(2.0, float(timeout))

    async def lookup(self, city: str) -> dict[str, Any]:
        city = " ".join(str(city or "").split()).strip()
        if not city:
            return {"ok": False, "needs_location": True, "reason": "no_city"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                geo = await client.get(
                    _GEOCODE_URL,
                    params={"name": city, "count": 1, "language": "zh"},
                )
                geo.raise_for_status()
                results = geo.json().get("results") or []
                if not results:
                    return {"ok": False, "needs_location": True,
                            "reason": "city_not_found", "city": city}
                place = results[0]
                forecast = await client.get(
                    _FORECAST_URL,
                    params={
                        "latitude": place.get("latitude"),
                        "longitude": place.get("longitude"),
                        "current": "temperature_2m,weather_code",
                        "daily": ("temperature_2m_max,temperature_2m_min,"
                                  "precipitation_probability_max"),
                        "timezone": "auto",
                        "forecast_days": 2,
                    },
                )
                forecast.raise_for_status()
                data = forecast.json()
        except Exception as exc:
            return {"ok": False, "reason": type(exc).__name__, "city": city}

        current = data.get("current") or {}
        daily = data.get("daily") or {}
        days = list(daily.get("time") or [])
        return {
            "ok": True,
            "city": place.get("name") or city,
            "admin": place.get("admin1") or "",
            "country": place.get("country") or "",
            "timezone": data.get("timezone") or "",
            "observed_at": current.get("time") or "",
            "current_c": current.get("temperature_2m"),
            "condition": describe_code(current.get("weather_code")),
            "today": _day(daily, 0, days),
            "tomorrow": _day(daily, 1, days),
            "source": "open-meteo",
        }


def _day(daily: dict[str, Any], index: int, days: list[Any]) -> dict[str, Any] | None:
    try:
        return {
            "date": days[index],
            "min_c": (daily.get("temperature_2m_min") or [])[index],
            "max_c": (daily.get("temperature_2m_max") or [])[index],
            "precip_prob": (daily.get("precipitation_probability_max") or [])[index],
        }
    except Exception:
        return None


def format_evidence(result: dict[str, Any]) -> str:
    """把查询结果压成一条可注入 prompt 的证据（不是给用户的台词）。"""
    if not result.get("ok"):
        if result.get("needs_location"):
            return "天气查询：用户没有说城市。先自然问他在哪个城市，不要编天气。"
        return "天气查询失败：本轮没有可用的实时天气数据，不要编具体数值。"
    parts = [f"{result.get('city')} {result.get('condition')} "
             f"{result.get('current_c')}°C"]
    today = result.get("today") or {}
    if today:
        parts.append(f"今天 {today.get('min_c')}~{today.get('max_c')}°C，"
                     f"降水概率 {today.get('precip_prob')}%")
    tomorrow = result.get("tomorrow") or {}
    if tomorrow:
        parts.append(f"明天 {tomorrow.get('min_c')}~{tomorrow.get('max_c')}°C，"
                     f"降水概率 {tomorrow.get('precip_prob')}%")
    return "实时天气（Open-Meteo，" + str(result.get("observed_at")) + "）：" + "；".join(parts) + "。"
