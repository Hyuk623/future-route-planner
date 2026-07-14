from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import os
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

app = FastAPI(title="MiriGil ML Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ============================================================
# Korean Holiday Calendar
# ============================================================
KOREAN_HOLIDAYS = {
    "2025-01-01": "new_year",
    "2025-01-28": "seollal", "2025-01-29": "seollal", "2025-01-30": "seollal",
    "2025-03-01": "independence",
    "2025-05-05": "children", "2025-05-06": "substitute",
    "2025-05-15": "buddha",
    "2025-06-06": "memorial",
    "2025-08-15": "liberation",
    "2025-10-03": "gaecheonjeol",
    "2025-10-05": "chuseok", "2025-10-06": "chuseok", "2025-10-07": "chuseok",
    "2025-10-08": "substitute",
    "2025-10-09": "hangul",
    "2025-12-25": "christmas",
    "2026-01-01": "new_year",
    "2026-02-16": "seollal", "2026-02-17": "seollal", "2026-02-18": "seollal",
    "2026-03-01": "independence", "2026-03-02": "substitute",
    "2026-05-05": "children",
    "2026-06-03": "buddha",
    "2026-06-06": "memorial",
    "2026-07-17": "constitution",
    "2026-08-15": "liberation",
    "2026-08-17": "substitute",
    "2026-09-24": "chuseok", "2026-09-25": "chuseok", "2026-09-26": "chuseok",
    "2026-10-03": "gaecheonjeol",
    "2026-10-05": "substitute",
    "2026-10-09": "hangul",
    "2026-12-25": "christmas",
    "2027-01-01": "new_year",
    "2027-02-06": "seollal", "2027-02-07": "seollal", "2027-02-08": "seollal",
    "2027-02-09": "substitute",
    "2027-03-01": "independence",
    "2027-05-05": "children",
    "2027-05-13": "buddha",
    "2027-06-06": "memorial",
    "2027-08-15": "liberation", "2027-08-16": "substitute",
    "2027-10-03": "gaecheonjeol", "2027-10-04": "substitute",
    "2027-10-09": "hangul",
    "2027-10-13": "chuseok", "2027-10-14": "chuseok", "2027-10-15": "chuseok",
    "2027-12-25": "christmas",
}

MAJOR_HOLIDAYS = {"seollal", "chuseok"}

HOLIDAY_NAMES_KR = {
    "new_year": "신정",
    "seollal": "설날",
    "independence": "삼일절",
    "children": "어린이날",
    "substitute": "대체 공휴일",
    "buddha": "부처님오신날",
    "memorial": "현충일",
    "liberation": "광복절",
    "gaecheonjeol": "개천절",
    "chuseok": "추석",
    "hangul": "한글날",
    "christmas": "성탄절",
    "constitution": "제헌절",
}


def get_holiday_features(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    is_holiday = date_str in KOREAN_HOLIDAYS
    holiday_type = KOREAN_HOLIDAYS.get(date_str, "none")
    is_major_holiday = holiday_type in MAJOR_HOLIDAYS

    is_pre_holiday = False
    is_post_holiday = False
    for offset in range(1, 3):
        future = (dt + timedelta(days=offset)).strftime("%Y-%m-%d")
        past = (dt - timedelta(days=offset)).strftime("%Y-%m-%d")
        if future in KOREAN_HOLIDAYS:
            is_pre_holiday = True
        if past in KOREAN_HOLIDAYS:
            is_post_holiday = True

    is_bridge_day = False
    if dt.weekday() < 5 and not is_holiday:
        prev = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        next_day = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        prev_is_off = (dt - timedelta(days=1)).weekday() >= 5 or prev in KOREAN_HOLIDAYS
        next_is_off = (dt + timedelta(days=1)).weekday() >= 5 or next_day in KOREAN_HOLIDAYS
        if prev_is_off and next_is_off:
            is_bridge_day = True

    return {
        "is_holiday": int(is_holiday),
        "is_major_holiday": int(is_major_holiday),
        "is_pre_holiday": int(is_pre_holiday),
        "is_post_holiday": int(is_post_holiday),
        "is_bridge_day": int(is_bridge_day),
    }


# ============================================================
# Model Load
# ============================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_traffic_model.json")
META_PATH = os.path.join(BASE_DIR, "model_meta.json")

model = xgb.XGBRegressor()
feature_cols = []

if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)
    print("[SUCCESS] XGBoost model loaded.")
else:
    print("[WARNING] Model not found. Run train_model.py first.")

if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
        feature_cols = meta.get("feature_cols", [])
    print(f"[SUCCESS] Metadata loaded. Features: {len(feature_cols)}")


# ============================================================
# Request / Response
# ============================================================
class PredictionRequest(BaseModel):
    base_duration: float
    distance: float
    precip_prob: float
    date_str: str
    time_str: str
    mode: str


@app.post("/predict")
async def predict_delay(req: PredictionRequest):
    try:
        dt = datetime.strptime(req.date_str, "%Y-%m-%d")
        hour = int(req.time_str.split(":")[0]) if req.time_str else 12
    except Exception:
        dt = datetime.now()
        hour = 12

    date_str = req.date_str
    day_of_week = dt.weekday()
    month = dt.month

    is_rain = 1 if req.precip_prob > 50 else 0
    is_heavy_rain = 1 if req.precip_prob > 80 else 0
    is_weekend = int(day_of_week >= 5)
    is_friday = int(day_of_week == 4)

    is_morning_rush = int(7 <= hour <= 9)
    is_evening_rush = int(17 <= hour <= 19)
    is_late_night = int(22 <= hour or hour <= 5)

    holiday_feat = get_holiday_features(date_str)

    is_summer_vacation = int(month in [7, 8])
    is_spring_blossom = int(month in [3, 4])
    is_autumn_foliage = int(month in [10, 11])
    is_year_end = int(month == 12)

    features = pd.DataFrame([{
        "base_duration": req.base_duration,
        "distance": req.distance,
        "precip_prob": req.precip_prob,
        "is_rain": is_rain,
        "is_heavy_rain": is_heavy_rain,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_friday": is_friday,
        "hour": hour,
        "is_morning_rush": is_morning_rush,
        "is_evening_rush": is_evening_rush,
        "is_late_night": is_late_night,
        **holiday_feat,
        "is_summer_vacation": is_summer_vacation,
        "is_spring_blossom": is_spring_blossom,
        "is_autumn_foliage": is_autumn_foliage,
        "is_year_end": is_year_end,
    }])

    if feature_cols:
        features = features[feature_cols]

    predicted_delay = float(model.predict(features)[0])
    if predicted_delay < 0:
        predicted_delay = 0

    if req.mode in ["cycling", "walking"]:
        if is_rain:
            predicted_delay = req.base_duration * 0.35
        else:
            predicted_delay = max(0, predicted_delay * 0.2)

    predicted_total = req.base_duration + predicted_delay

    # ============================================================
    # 맥락 인식형(Context-Aware) 영향 변수 분석
    # ============================================================
    # 핵심 원칙:
    # 1. 공휴일/연휴에는 출퇴근 러시아워가 적용되지 않는다
    # 2. 귀성/귀경 정체는 주로 오후~저녁에 발생한다
    # 3. 공휴일 이른 아침은 오히려 한산하다
    # 4. 여러 변수가 겹칠 경우 가장 지배적인 요인을 우선 표시한다
    # ============================================================

    impacts = []
    holiday_type = KOREAN_HOLIDAYS.get(date_str, None)
    is_off_day = bool(holiday_feat["is_holiday"]) or bool(is_weekend)

    # 시간대 구분
    is_early_morning = (hour <= 6)
    is_morning = (7 <= hour <= 11)
    is_afternoon = (12 <= hour <= 16)
    is_evening = (17 <= hour <= 21)
    is_nighttime = (22 <= hour or hour <= 5)

    # ── 1. 명절(설날/추석) 당일 ──────────────────────────────────
    if holiday_feat["is_major_holiday"]:
        name = HOLIDAY_NAMES_KR.get(holiday_type, "명절")
        if is_morning:
            impacts.append({
                "type": "traffic", "icon": "fa-car-burst", "iconColor": "icon-traffic",
                "title": f"{name} 연휴 귀성길 정체 절정",
                "desc": f"{name} 당일 오전은 마지막 귀성 차량이 집중되어 고속도로가 극심하게 막힙니다.",
                "result": "대규모 지연 반영됨",
            })
        elif is_afternoon or is_evening:
            impacts.append({
                "type": "traffic", "icon": "fa-car-burst", "iconColor": "icon-traffic",
                "title": f"{name} 연휴 나들이/귀경 혼잡",
                "desc": f"{name} 당일 오후~저녁은 가족 이동과 조기 귀경 차량이 겹쳐 전국 도로가 혼잡합니다.",
                "result": "대규모 지연 반영됨",
            })
        else:
            impacts.append({
                "type": "info", "icon": "fa-moon", "iconColor": "icon-info",
                "title": f"{name} 연휴이나 새벽/심야라 도로 비교적 한산",
                "desc": "명절 당일이지만 새벽/심야 시간대로 이동 차량이 적어 비교적 원활합니다.",
                "result": "소폭 지연 반영됨",
            })

    # ── 2. 귀성길 (연휴 전날) ────────────────────────────────────
    elif holiday_feat["is_pre_holiday"]:
        if is_afternoon or is_evening:
            severity = "절정" if is_evening else "시작"
            impacts.append({
                "type": "traffic", "icon": "fa-road", "iconColor": "icon-traffic",
                "title": f"연휴 전날 귀성길 정체 {severity}",
                "desc": "연휴를 앞두고 귀성 차량이 집중되는 시간대입니다. 고속도로 진입 전 여유 시간을 두세요.",
                "result": "귀성길 지연 반영됨",
            })
        elif is_morning:
            impacts.append({
                "type": "traffic", "icon": "fa-road", "iconColor": "icon-traffic",
                "title": "연휴 전날 오전 — 귀성 이동 시작",
                "desc": "연휴 전날이지만 오전은 아직 귀성 차량이 많지 않습니다. 오후부터 급격히 혼잡해집니다.",
                "result": "소폭 지연 반영됨",
            })
        else:
            impacts.append({
                "type": "info", "icon": "fa-moon", "iconColor": "icon-info",
                "title": "연휴 전날 새벽/심야 — 도로 한산",
                "desc": "귀성 차량 대부분이 아직 이동 전이라 도로가 비교적 원활합니다.",
                "result": "거의 정상 소요시간",
            })

    # ── 3. 귀경길 (연휴 다음날) ──────────────────────────────────
    elif holiday_feat["is_post_holiday"]:
        if is_afternoon or is_evening:
            severity = "절정" if is_evening else "시작"
            impacts.append({
                "type": "traffic", "icon": "fa-road", "iconColor": "icon-traffic",
                "title": f"연휴 후 귀경길 정체 {severity}",
                "desc": "연휴가 끝나고 복귀하는 차량이 몰리는 시간대입니다. 서울 방면 도로가 특히 혼잡합니다.",
                "result": "귀경길 지연 반영됨",
            })
        elif is_morning:
            impacts.append({
                "type": "info", "icon": "fa-road", "iconColor": "icon-info",
                "title": "귀경 이동 시작 (오전 — 비교적 원활)",
                "desc": "귀경길 정체는 주로 오후부터 시작됩니다. 오전 출발을 권장합니다.",
                "result": "소폭 지연 반영됨",
            })
        else:
            impacts.append({
                "type": "info", "icon": "fa-moon", "iconColor": "icon-info",
                "title": "귀경 피크 이후 (심야 — 도로 완화)",
                "desc": "귀경 차량 대부분이 이미 이동을 완료한 시간대로 도로가 완화되고 있습니다.",
                "result": "지연 완화됨",
            })

    # ── 4. 일반 공휴일 ────────────────────────────────────────────
    elif holiday_feat["is_holiday"]:
        name = HOLIDAY_NAMES_KR.get(holiday_type, "공휴일")
        if is_early_morning or (is_morning and hour < 9):
            impacts.append({
                "type": "info", "icon": "fa-sun", "iconColor": "icon-info",
                "title": f"{name} — 이른 아침, 도로 한산",
                "desc": "공휴일 이른 아침은 출근 차량이 없어 오히려 평일보다 도로가 한산합니다.",
                "result": "평일 대비 원활한 흐름",
            })
        elif is_afternoon or is_evening:
            impacts.append({
                "type": "traffic", "icon": "fa-calendar-day", "iconColor": "icon-traffic",
                "title": f"{name} 오후 — 나들이/귀가 차량 증가",
                "desc": "공휴일 오후는 나들이를 즐기고 귀가하는 차량이 몰려 주요 도로가 혼잡해집니다.",
                "result": "공휴일 귀가 지연 반영됨",
            })
        else:
            impacts.append({
                "type": "info", "icon": "fa-calendar-check", "iconColor": "icon-info",
                "title": f"{name} 공휴일 (비교적 원활한 시간대)",
                "desc": "공휴일이지만 현재 시간대는 이동 차량이 많지 않아 비교적 원활합니다.",
                "result": "소폭 지연 반영됨",
            })

    # ── 5. 징검다리 연휴 (별도 추가) ─────────────────────────────
    if holiday_feat["is_bridge_day"]:
        impacts.append({
            "type": "traffic", "icon": "fa-bridge", "iconColor": "icon-traffic",
            "title": "징검다리 연휴 — 도로 극혼잡 예상",
            "desc": "공휴일과 주말 사이에 낀 평일로 많은 사람이 연차를 사용합니다. 여행 및 귀가 차량이 급증합니다.",
            "result": "징검다리 연휴 지연 반영됨",
        })

    # ── 6. 출퇴근 러시아워 — 평일(비출근일 제외)에만 적용 ──────────
    if not is_off_day:
        if is_morning_rush:
            if is_friday:
                impacts.append({
                    "type": "traffic", "icon": "fa-clock", "iconColor": "icon-event",
                    "title": "금요일 출근 시간대 (07~09시) — 복합 정체",
                    "desc": "금요일 아침은 출근 차량과 주말 이동 차량이 섞여 평소보다 더 혼잡합니다.",
                    "result": "복합 러시아워 지연 반영됨",
                })
            else:
                impacts.append({
                    "type": "traffic", "icon": "fa-clock", "iconColor": "icon-event",
                    "title": "출근 시간대 (07~09시) 정체",
                    "desc": "출근 러시아워로 도심 진입 구간에 심한 정체가 예상됩니다.",
                    "result": "러시아워 지연 반영됨",
                })
        elif is_evening_rush:
            if is_friday:
                impacts.append({
                    "type": "traffic", "icon": "fa-clock", "iconColor": "icon-traffic",
                    "title": "금요일 퇴근 시간대 (17~19시) — 주말 이동 복합 정체",
                    "desc": "금요일 저녁은 퇴근 차량과 주말 여행·귀가 차량이 겹쳐 한 주 중 가장 혼잡한 시간대입니다.",
                    "result": "복합 최대 정체 반영됨",
                })
            else:
                impacts.append({
                    "type": "traffic", "icon": "fa-clock", "iconColor": "icon-event",
                    "title": "퇴근 시간대 (17~19시) 정체",
                    "desc": "퇴근 러시아워로 도심 및 외곽 연결 도로에 심한 정체가 예상됩니다.",
                    "result": "러시아워 지연 반영됨",
                })
    else:
        # 공휴일/주말이지만 연휴 관련 항목이 이미 없는 저녁 → 주말 귀가 정체
        no_holiday_impact = not (
            holiday_feat["is_holiday"] or holiday_feat["is_pre_holiday"] or
            holiday_feat["is_post_holiday"] or holiday_feat["is_major_holiday"]
        )
        if is_evening and no_holiday_impact:
            impacts.append({
                "type": "traffic", "icon": "fa-car", "iconColor": "icon-event",
                "title": "주말 저녁 귀가 차량 증가",
                "desc": "주말을 즐기고 귀가하는 차량이 몰려 주요 도심 진입 도로가 혼잡합니다.",
                "result": "주말 귀가 정체 반영됨",
            })

    # ── 7. 금요일 오후 추가 안내 ──────────────────────────────────
    if is_friday and not is_off_day and not is_morning_rush and not is_evening_rush and is_afternoon:
        impacts.append({
            "type": "traffic", "icon": "fa-calendar-week", "iconColor": "icon-event",
            "title": "금요일 오후 — 주말 이동 차량 증가 시작",
            "desc": "금요일 오후부터 주말 여행·귀가 차량이 몰리기 시작합니다.",
            "result": "금요일 오후 지연 반영됨",
        })

    # ── 8. 기상 조건 (항상 적용) ──────────────────────────────────
    if is_heavy_rain:
        impacts.append({
            "type": "weather", "icon": "fa-cloud-showers-heavy", "iconColor": "icon-weather",
            "title": f"폭우 예보 (강수 확률 {int(req.precip_prob)}%)",
            "desc": "강한 비로 인해 시야 저하, 노면 미끄러움, 감속 운행이 예상됩니다. 안전 거리를 충분히 확보하세요.",
            "result": "기상 악화 지연 반영됨",
        })
    elif is_rain:
        impacts.append({
            "type": "weather", "icon": "fa-cloud-rain", "iconColor": "icon-weather",
            "title": f"비 예보 (강수 확률 {int(req.precip_prob)}%)",
            "desc": "비로 인해 노면이 미끄럽고 감속 운행이 예상됩니다.",
            "result": "기상 지연 반영됨",
        })

    # ── 9. 계절 효과 — 교통 정체 요인 없을 때만 보조 표시 ─────────
    traffic_impacts = [i for i in impacts if i["type"] == "traffic"]
    if len(traffic_impacts) == 0 and is_afternoon:
        if is_summer_vacation:
            impacts.append({
                "type": "event", "icon": "fa-umbrella-beach", "iconColor": "icon-event",
                "title": "여름 휴가철 교통량 증가",
                "desc": "7~8월 여름 휴가 시즌 오후로 해수욕장, 관광지 방면 교통량이 증가합니다.",
                "result": "시즌 효과 반영됨",
            })
        elif is_autumn_foliage:
            impacts.append({
                "type": "event", "icon": "fa-leaf", "iconColor": "icon-event",
                "title": "가을 단풍철 관광 차량 증가",
                "desc": "단풍 시즌 오후로 산간 및 관광지 주변 도로가 혼잡합니다.",
                "result": "시즌 효과 반영됨",
            })

    # ── 10. 아무 정체 요인 없음 ───────────────────────────────────
    if len(impacts) == 0:
        if is_nighttime:
            impacts.append({
                "type": "info", "icon": "fa-moon", "iconColor": "icon-info",
                "title": "심야 시간대 — 쾌적한 주행 예상",
                "desc": "심야 시간대로 교통량이 매우 적어 빠르게 이동할 수 있습니다.",
                "result": "정상 소요시간 (심야 원활)",
            })
        else:
            impacts.append({
                "type": "info", "icon": "fa-check", "iconColor": "icon-info",
                "title": "원활한 교통 흐름 예상",
                "desc": "해당 날짜와 시간대에 특별한 정체 요인이 없습니다. 쾌적한 주행이 예상됩니다.",
                "result": "정상 소요시간 예상",
            })

    return {
        "status": "success",
        "predicted_total_mins": round(predicted_total),
        "predicted_delay_mins": round(predicted_delay),
        "impacts": impacts,
    }


@app.get("/")
def root():
    return {"message": "MiriGil ML Backend is running!"}


# ============================================================
# Kakao API Proxy (CORS 우회용)
# ============================================================
KAKAO_API_KEY = "0a7df2bc6a679d32af81d6897e0e9480"

@app.get("/proxy/search")
def proxy_search(query: str):
    url = f"https://dapi.kakao.com/v2/local/search/keyword.json?query={urllib.parse.quote(query)}&size=7"
    req = urllib.request.Request(url, headers={"Authorization": f"KakaoAK {KAKAO_API_KEY}"})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}

@app.get("/proxy/reverse")
def proxy_reverse(lat: str, lon: str):
    url = f"https://dapi.kakao.com/v2/local/geo/coord2address.json?x={lon}&y={lat}"
    req = urllib.request.Request(url, headers={"Authorization": f"KakaoAK {KAKAO_API_KEY}"})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}

@app.get("/proxy/route")
def proxy_route(origin: str, destination: str, priority: str = "RECOMMEND", alternatives: str = "false"):
    # origin/destination format: lon,lat
    url = f"https://apis-navi.kakaomobility.com/v1/directions?origin={origin}&destination={destination}&priority={priority}&alternatives={alternatives}"
    req = urllib.request.Request(url, headers={"Authorization": f"KakaoAK {KAKAO_API_KEY}"})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
