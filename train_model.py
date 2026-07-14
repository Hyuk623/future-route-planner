"""
미리길 (Future Route Planner) - 교통 정체 예측 ML 모델 학습 스크립트
====================================================================
한국 도로교통 정체의 핵심 요인들을 종합 분석하여 XGBoost 모델에 학습시킵니다.

[교통 정체 핵심 요인 분석]
1. 공휴일/연휴 효과: 명절(설날, 추석) 귀성/귀경길, 징검다리 연휴 전후
2. 요일 효과: 금요일 저녁 > 월요일 아침 > 주말 > 평일
3. 시간대 효과: 출퇴근 러시아워(07-09시, 17-19시)
4. 계절/시즌 효과: 여름 휴가철(7-8월), 봄꽃/가을 단풍 시즌
5. 기상 조건: 강수 확률, 강설, 안개
6. 거리/경로 특성: 장거리일수록 정체 구간 조우 확률 증가
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import json

print("=" * 60)
print("  MiriGil ML Pipeline - Traffic Delay Prediction Model")
print("=" * 60)

# ============================================================
# 1. 한국 공휴일/연휴 캘린더 (2025-2027)
# ============================================================
# 실제 서비스에서는 공공데이터포털(data.go.kr)의
# '특일 정보' API를 활용하여 실시간으로 가져올 수 있습니다.
KOREAN_HOLIDAYS = {
    # 2025년
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
    # 2026년
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
    # 2027년
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

# 명절(설날/추석)은 귀성길/귀경길 특별 처리 대상
MAJOR_HOLIDAYS = {"seollal", "chuseok"}

def get_holiday_features(date_str):
    """주어진 날짜의 공휴일/연휴 관련 피처를 계산합니다."""
    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str, "%Y-%m-%d")

    is_holiday = date_str in KOREAN_HOLIDAYS
    holiday_type = KOREAN_HOLIDAYS.get(date_str, "none")

    # 명절인지 확인
    is_major_holiday = holiday_type in MAJOR_HOLIDAYS

    # 명절/연휴 전후 귀성/귀경길 판단 (전후 2일)
    is_pre_holiday = False  # 귀성길 (연휴 직전)
    is_post_holiday = False  # 귀경길 (연휴 직후)

    for offset in range(1, 3):
        future = (dt + timedelta(days=offset)).strftime("%Y-%m-%d")
        past = (dt - timedelta(days=offset)).strftime("%Y-%m-%d")
        if future in KOREAN_HOLIDAYS:
            is_pre_holiday = True
        if past in KOREAN_HOLIDAYS:
            is_post_holiday = True

    # 징검다리 연휴 판단: 공휴일과 주말 사이에 낀 평일
    is_bridge_day = False
    if dt.weekday() < 5 and not is_holiday:  # 평일이고 공휴일이 아닐 때
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


def get_time_features(hour):
    """시간대별 교통 특성을 피처로 변환합니다."""
    is_morning_rush = int(7 <= hour <= 9)
    is_evening_rush = int(17 <= hour <= 19)
    is_late_night = int(22 <= hour or hour <= 5)
    return {
        "is_morning_rush": is_morning_rush,
        "is_evening_rush": is_evening_rush,
        "is_late_night": is_late_night,
    }


def get_season_features(month):
    """계절/시즌 효과를 피처로 변환합니다."""
    is_summer_vacation = int(month in [7, 8])
    is_spring_blossom = int(month in [3, 4])
    is_autumn_foliage = int(month in [10, 11])
    is_year_end = int(month == 12)
    return {
        "is_summer_vacation": is_summer_vacation,
        "is_spring_blossom": is_spring_blossom,
        "is_autumn_foliage": is_autumn_foliage,
        "is_year_end": is_year_end,
    }


# ============================================================
# 2. 대규모 가상 교통 데이터 생성 (10만 건)
# ============================================================
print("\n[1/4] 10만 건의 종합 교통 빅데이터를 생성 중...")
np.random.seed(42)
NUM_SAMPLES = 100000

# 날짜 범위: 2025-01-01 ~ 2027-12-31 (약 3년)
date_range = pd.date_range("2025-01-01", "2027-12-31", freq="D")
records = []

for _ in range(NUM_SAMPLES):
    # 랜덤 날짜 선택
    raw_date = np.random.choice(date_range)
    date = pd.to_datetime(raw_date)
    date_str = date.strftime("%Y-%m-%d")
    day_of_week = date.dayofweek  # 0=Mon, 6=Sun
    month = date.month
    hour = np.random.randint(0, 24)

    # 기본 경로 정보
    base_duration = np.random.randint(10, 360)  # 10분 ~ 6시간
    distance = base_duration * np.random.uniform(0.7, 1.3)

    # 기상 조건
    precip_prob = np.random.randint(0, 100)
    is_rain = int(precip_prob > 50)
    is_heavy_rain = int(precip_prob > 80)

    # 피처 계산
    holiday_feat = get_holiday_features(date_str)
    time_feat = get_time_features(hour)
    season_feat = get_season_features(month)

    is_weekend = int(day_of_week >= 5)
    is_friday = int(day_of_week == 4)

    # ========================================
    # 지연 시간 계산 (실제 도로교통 패턴 반영)
    # ========================================
    delay = base_duration * 0.03  # 기본 3% 지연

    # [요일 효과]
    if is_friday:
        delay += base_duration * 0.12  # 금요일: +12%
    elif is_weekend:
        delay += base_duration * 0.05  # 주말: +5%

    # [시간대 효과]
    if time_feat["is_morning_rush"]:
        delay += base_duration * 0.18  # 출근 러시: +18%
    if time_feat["is_evening_rush"]:
        delay += base_duration * 0.22  # 퇴근 러시: +22%
    if time_feat["is_late_night"]:
        delay -= base_duration * 0.05  # 심야: -5% (도로 한산)

    # [공휴일/연휴 효과]
    if holiday_feat["is_major_holiday"]:
        delay += np.random.randint(40, 120)  # 설/추석 당일: +40~120분
    elif holiday_feat["is_holiday"]:
        delay += np.random.randint(15, 50)  # 일반 공휴일: +15~50분

    if holiday_feat["is_pre_holiday"]:
        delay += np.random.randint(30, 90)  # 귀성길: +30~90분
    if holiday_feat["is_post_holiday"]:
        delay += np.random.randint(25, 80)  # 귀경길: +25~80분
    if holiday_feat["is_bridge_day"]:
        delay += np.random.randint(20, 60)  # 징검다리 연휴: +20~60분

    # [계절 효과]
    if season_feat["is_summer_vacation"]:
        delay += base_duration * 0.08  # 여름 휴가철: +8%
    if season_feat["is_autumn_foliage"]:
        delay += base_duration * 0.06  # 가을 단풍철: +6%
    if season_feat["is_year_end"]:
        delay += base_duration * 0.10  # 연말: +10%

    # [기상 효과]
    if is_rain:
        delay += base_duration * 0.15  # 비: +15%
    if is_heavy_rain:
        delay += base_duration * 0.10  # 폭우 추가: +10%

    # 노이즈 추가
    delay += np.random.normal(0, 3)
    delay = max(0, int(delay))

    record = {
        "base_duration": base_duration,
        "distance": round(distance, 1),
        "precip_prob": precip_prob,
        "is_rain": is_rain,
        "is_heavy_rain": is_heavy_rain,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_friday": is_friday,
        "hour": hour,
        **time_feat,
        **holiday_feat,
        **season_feat,
        "delay_minutes": delay,
    }
    records.append(record)

df = pd.DataFrame(records)

FEATURE_COLS = [
    "base_duration", "distance", "precip_prob", "is_rain", "is_heavy_rain",
    "day_of_week", "is_weekend", "is_friday", "hour",
    "is_morning_rush", "is_evening_rush", "is_late_night",
    "is_holiday", "is_major_holiday", "is_pre_holiday", "is_post_holiday", "is_bridge_day",
    "is_summer_vacation", "is_spring_blossom", "is_autumn_foliage", "is_year_end",
]

print(f"   -> 생성 완료: {len(df)} 건, 피처 {len(FEATURE_COLS)}개")
print(f"   -> 피처 목록: {FEATURE_COLS}")

# ============================================================
# 3. 모델 학습 (XGBoost)
# ============================================================
print("\n[2/4] 학습/검증 데이터 분할 중...")
X = df[FEATURE_COLS]
y = df["delay_minutes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   -> Train: {len(X_train)}, Test: {len(X_test)}")

print("\n[3/4] XGBoost 모델 학습 중...")
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# ============================================================
# 4. 성능 평가
# ============================================================
print("\n[4/4] 모델 성능 평가...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"   -> RMSE: {rmse:.2f}분 (평균 제곱근 오차)")
print(f"   -> MAE:  {mae:.2f}분 (평균 절대 오차)")

# 피처 중요도 출력
importance = model.feature_importances_
feat_importance = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)
print("\n   [피처 중요도 Top 10]")
for feat, imp in feat_importance[:10]:
    bar = "#" * int(imp * 100)
    print(f"   {feat:25s} {imp:.4f} {bar}")

# ============================================================
# 5. 모델 및 메타데이터 저장
# ============================================================
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "xgboost_traffic_model.json")
model.save_model(model_path)
print(f"\n   -> Model saved: {model_path}")

# 피처 목록 저장 (서버에서 로드할 때 사용)
meta = {"feature_cols": FEATURE_COLS}
meta_path = os.path.join(base_dir, "model_meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"   -> Metadata saved: {meta_path}")

print("\n" + "=" * 60)
print("  Pipeline complete!")
print("=" * 60)
