// Configuration
const OSRM_API = 'https://router.project-osrm.org/route/v1';
const METEO_API = 'https://api.open-meteo.com/v1/forecast';
const ML_BACKEND = '/predict';
const KAKAO_SEARCH_PROXY = '/proxy/search';
const KAKAO_REVERSE_PROXY = '/proxy/reverse';

// State
let map;
let routeLayer;
let startMarker;
let endMarker;
let startCoords = null;
let endCoords = null;
let currentMode = 'driving';
let currentRouteOpt = 'recommended';
let mapClickTarget = null; // 'start' or 'end' — for map click selection

// Custom Marker Icons
const startIcon = L.divIcon({
    html: '<i class="fa-solid fa-location-dot" style="color:#3b82f6;font-size:28px;filter:drop-shadow(0 2px 4px rgba(0,0,0,0.5))"></i>',
    iconSize: [28, 28],
    iconAnchor: [14, 28],
    className: 'custom-marker'
});
const endIcon = L.divIcon({
    html: '<i class="fa-solid fa-location-dot" style="color:#ef4444;font-size:28px;filter:drop-shadow(0 2px 4px rgba(0,0,0,0.5))"></i>',
    iconSize: [28, 28],
    iconAnchor: [14, 28],
    className: 'custom-marker'
});

// DOM Elements
const startInput = document.getElementById('start-input');
const endInput = document.getElementById('end-input');
const startSuggestions = document.getElementById('start-suggestions');
const endSuggestions = document.getElementById('end-suggestions');
const dateInput = document.getElementById('date-input');
const timeInput = document.getElementById('time-input');
const predictBtn = document.getElementById('predict-btn');
const resultsPanel = document.getElementById('results-panel');
const loadingOverlay = document.getElementById('loading-overlay');
const impactList = document.getElementById('impact-list');
const tabBtns = document.querySelectorAll('.tab-btn');
const optionChips = document.querySelectorAll('.option-chip');
const mapTooltip = document.getElementById('map-tooltip');

// Initialize App
function init() {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    dateInput.value = tomorrow.toISOString().split('T')[0];
    timeInput.value = "09:00";

    map = L.map('map', { zoomControl: false }).setView([37.5665, 126.9780], 11);
    L.control.zoom({ position: 'bottomright' }).addTo(map);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    setupEventListeners();
}

// Event Listeners
function setupEventListeners() {
    startInput.addEventListener('input', debounce((e) => searchLocation(e.target.value, 'start'), 400));
    endInput.addEventListener('input', debounce((e) => searchLocation(e.target.value, 'end'), 400));

    // Focus events — activate map click target
    startInput.addEventListener('focus', () => {
        mapClickTarget = 'start';
        mapTooltip.innerHTML = '<i class="fa-solid fa-map-pin"></i> 지도를 클릭하면 <b>출발지</b>로 설정됩니다';
        mapTooltip.classList.remove('hidden');
    });
    endInput.addEventListener('focus', () => {
        mapClickTarget = 'end';
        mapTooltip.innerHTML = '<i class="fa-solid fa-map-pin"></i> 지도를 클릭하면 <b>도착지</b>로 설정됩니다';
        mapTooltip.classList.remove('hidden');
    });

    document.addEventListener('click', (e) => {
        if (e.target !== startInput) startSuggestions.classList.add('hidden');
        if (e.target !== endInput) endSuggestions.classList.add('hidden');
    });

    predictBtn.addEventListener('click', handlePrediction);

    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            tabBtns.forEach(b => b.classList.remove('active'));
            const targetBtn = e.target.closest('.tab-btn');
            targetBtn.classList.add('active');
            currentMode = targetBtn.dataset.mode;

            const inputsDiv = document.querySelector('.location-inputs');
            const dateDiv = document.querySelector('.datetime-inputs');
            const routeOptionsDiv = document.getElementById('route-options-container');
            const predictBtn = document.getElementById('predict-btn');
            const devWarning = document.getElementById('dev-warning');

            if (currentMode !== 'driving') {
                inputsDiv.classList.add('hidden');
                dateDiv.classList.add('hidden');
                routeOptionsDiv.classList.add('hidden');
                predictBtn.classList.add('hidden');
                resultsPanel.classList.add('hidden');
                devWarning.classList.remove('hidden');
                if (routeLayer) map.removeLayer(routeLayer);
            } else {
                inputsDiv.classList.remove('hidden');
                dateDiv.classList.remove('hidden');
                routeOptionsDiv.classList.remove('hidden');
                predictBtn.classList.remove('hidden');
                devWarning.classList.add('hidden');
            }
        });
    });

    // Route Options Click
    optionChips.forEach(chip => {
        chip.addEventListener('click', (e) => {
            optionChips.forEach(c => c.classList.remove('active'));
            const targetChip = e.target.closest('.option-chip');
            targetChip.classList.add('active');
            currentRouteOpt = targetChip.dataset.routeOpt;
        });
    });

    // Map Click — select location by clicking on map
    map.on('click', async (e) => {
        if (!mapClickTarget) {
            // Default: if no input focused, auto-assign
            if (!startCoords) mapClickTarget = 'start';
            else if (!endCoords) mapClickTarget = 'end';
            else mapClickTarget = 'end'; // re-select end
        }

        const latlng = e.latlng;
        const coords = { lat: latlng.lat, lon: latlng.lng };

        // Reverse geocode via Kakao
        const address = await reverseGeocode(coords.lat, coords.lon);
        const displayName = address || `${coords.lat.toFixed(5)}, ${coords.lon.toFixed(5)}`;

        placeMarker(coords, displayName, mapClickTarget);

        // Auto-switch target
        if (mapClickTarget === 'start' && !endCoords) {
            mapClickTarget = 'end';
            mapTooltip.innerHTML = '<i class="fa-solid fa-map-pin"></i> 지도를 클릭하면 <b>도착지</b>로 설정됩니다';
        } else if (mapClickTarget === 'end' && !startCoords) {
            mapClickTarget = 'start';
            mapTooltip.innerHTML = '<i class="fa-solid fa-map-pin"></i> 지도를 클릭하면 <b>출발지</b>로 설정됩니다';
        }
    });
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => { clearTimeout(timeout); func(...args); };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ============================================================
// Kakao Local API — Geocoding (keyword search)
// ============================================================
async function searchLocation(query, type) {
    const suggestionsBox = type === 'start' ? startSuggestions : endSuggestions;

    if (!query || query.length < 2) {
        suggestionsBox.classList.add('hidden');
        return;
    }

    try {
        const response = await fetch(
            `${KAKAO_SEARCH_PROXY}?query=${encodeURIComponent(query)}`
        );
        const data = await response.json();

        suggestionsBox.innerHTML = '';
        if (data.documents && data.documents.length > 0) {
            data.documents.forEach(item => {
                const div = document.createElement('div');
                div.className = 'suggestion-item';
                // Show place name + address for clarity
                div.innerHTML = `
                    <strong>${item.place_name}</strong>
                    <span class="suggestion-address">${item.road_address_name || item.address_name}</span>
                `;
                div.addEventListener('click', () => {
                    const coords = { lat: parseFloat(item.y), lon: parseFloat(item.x) };
                    const name = `${item.place_name} (${item.road_address_name || item.address_name})`;
                    placeMarker(coords, name, type);
                    suggestionsBox.classList.add('hidden');
                });
                suggestionsBox.appendChild(div);
            });
            suggestionsBox.classList.remove('hidden');
        } else {
            suggestionsBox.classList.add('hidden');
        }
    } catch (error) {
        console.error('Kakao search error:', error);
    }
}

// Kakao Reverse Geocoding (coord -> address)
async function reverseGeocode(lat, lon) {
    try {
        const response = await fetch(
            `${KAKAO_REVERSE_PROXY}?lat=${lat}&lon=${lon}`
        );
        const data = await response.json();
        if (data.documents && data.documents.length > 0) {
            const doc = data.documents[0];
            return doc.road_address
                ? doc.road_address.address_name
                : doc.address.address_name;
        }
        return null;
    } catch (e) {
        console.error('Reverse geocode failed:', e);
        return null;
    }
}

// Place marker on map & update input
function placeMarker(coords, displayName, type) {
    if (type === 'start') {
        startInput.value = displayName;
        startCoords = coords;
        if (startMarker) map.removeLayer(startMarker);
        startMarker = L.marker([coords.lat, coords.lon], { icon: startIcon }).addTo(map);
    } else {
        endInput.value = displayName;
        endCoords = coords;
        if (endMarker) map.removeLayer(endMarker);
        endMarker = L.marker([coords.lat, coords.lon], { icon: endIcon }).addTo(map);
    }

    if (startCoords && endCoords) {
        const bounds = L.latLngBounds(
            [startCoords.lat, startCoords.lon],
            [endCoords.lat, endCoords.lon]
        );
        map.fitBounds(bounds, { padding: [50, 50] });
    }
}

// ============================================================
// Main Prediction Logic
// ============================================================
async function handlePrediction() {
    if (!startCoords || !endCoords) {
        alert("출발지와 도착지를 선택해주세요.\n\n검색어를 입력하거나, 지도를 직접 클릭하여 위치를 지정할 수 있습니다.");
        return;
    }
    if (!dateInput.value || !timeInput.value) {
        alert("출발 날짜와 시간을 선택해주세요.");
        return;
    }

    resultsPanel.classList.add('hidden');
    loadingOverlay.classList.remove('hidden');

    try {
        // 1. Fetch route from KakaoNavi
        const kakaoRoutes = await fetchKakaoRoute(startCoords, endCoords, currentRouteOpt);
        if (!kakaoRoutes || kakaoRoutes.length === 0) throw new Error("경로를 찾을 수 없습니다.");

        // 2. Select best matching alternative based on routeOpt
        const { simRoute, optImpact } = applyRouteOption(kakaoRoutes, currentRouteOpt);

        // 3. ML prediction
        const weatherData = await fetchWeather(endCoords, dateInput.value, timeInput.value);
        
        // Pass Kakao summary duration & distance to ML
        // Kakao duration is in seconds, ML expects minutes. distance is in meters, ML expects km.
        const baseRouteData = {
            duration: simRoute.summary.duration,
            distance: simRoute.summary.distance,
            sections: simRoute.sections
        };
        
        const impactAnalysis = await callMLBackend(baseRouteData, weatherData, currentMode, dateInput.value, timeInput.value);
        if (optImpact) impactAnalysis.impacts.unshift(optImpact);

        // 4. Compute congestion ratio for visualization
        const delayRatio = impactAnalysis.addedDurationSec / (baseRouteData.duration || 1);

        // 5. Render
        renderResults(baseRouteData, impactAnalysis, simRoute.summary.fare);
        drawKakaoCongestionRoute(simRoute.sections, delayRatio);
    } catch (error) {
        console.error(error);
        alert("경로 예측 중 오류가 발생했습니다: " + error.message);
    } finally {
        loadingOverlay.classList.add('hidden');
    }
}

// ============================================================
// KakaoNavi Routing API
// ============================================================
async function fetchKakaoRoute(start, end, routeOpt) {
    // Map our UI route options to Kakao API priorities
    let priority = "RECOMMEND";
    let alternatives = "false";
    
    if (routeOpt === 'fastest' || routeOpt === 'highway') priority = "TIME";
    if (routeOpt === 'shortest') priority = "DISTANCE";
    // For free or schoolzone, we ask for alternatives to try to find a better matching path
    if (routeOpt === 'free' || routeOpt === 'schoolzone') {
        priority = "RECOMMEND";
        alternatives = "true";
    }

    const origin = `${start.lon},${start.lat}`;
    const dest = `${end.lon},${end.lat}`;
    const url = `/proxy/route?origin=${origin}&destination=${dest}&priority=${priority}&alternatives=${alternatives}`;
    
    const response = await fetch(url);
    const data = await response.json();

    if (!data.routes || data.routes.length === 0) return null;
    return data.routes;
}

// ============================================================
// Route Option Selection: pick the best-fit route & apply weights
// OSRM route[0] = fastest (time-optimized)
// OSRM route[1] = 2nd alternative (usually different path)
// OSRM route[2] = 3rd alternative (longer, more varied)
// ============================================================
const ROUTE_OPT_CONFIG = {
    recommended: {
        icon: 'fa-star', color: 'icon-info',
        title: '카카오내비 추천 경로',
        desc: '시간과 거리를 종합적으로 고려한 최적 경로입니다.',
        result: '최적 균형 경로'
    },
    highway: {
        icon: 'fa-road', color: 'icon-info',
        title: '카카오내비 고속도로/간선 우선',
        desc: '고속도로·간선도로를 최대한 활용하여 쾌적하게 이동하는 경로입니다.',
        result: '고속도로 최우선 경유'
    },
    fastest: {
        icon: 'fa-gauge-high', color: 'icon-info',
        title: '카카오내비 최단 시간 경로',
        desc: '가장 빨리 도착할 수 있는 빠른 길입니다.',
        result: '소요 시간 최소화'
    },
    free: {
        icon: 'fa-coins', color: 'icon-info',
        title: '무료도로 우선 경로',
        desc: '통행료가 발생하지 않는 국도/지방도 위주의 대안 경로입니다.',
        result: '통행료 절약'
    },
    shortest: {
        icon: 'fa-ruler', color: 'icon-info',
        title: '카카오내비 최단 거리 경로',
        desc: '주행 거리가 가장 짧은 길입니다. 골목길이 포함될 수 있습니다.',
        result: '거리 최소화'
    },
    schoolzone: {
        icon: 'fa-child-reaching', color: 'text-warning',
        title: '대안 우회 경로 적용',
        desc: '추천 경로 외에 다른 길로 우회하는 대안 경로입니다.',
        result: '우회 경로 탐색'
    },
};

function applyRouteOption(routes, opt) {
    const cfg = ROUTE_OPT_CONFIG[opt] || ROUTE_OPT_CONFIG.recommended;
    let simRoute = routes[0];

    // If we asked for alternatives (free, schoolzone), try to find a specific one
    if (routes.length > 1) {
        if (opt === 'free') {
            // Find a route with 0 toll fare
            const freeRoute = routes.find(r => r.summary.fare.toll === 0);
            if (freeRoute) simRoute = freeRoute;
            else simRoute = routes[1]; // fallback to first alternative
        } else if (opt === 'schoolzone') {
            simRoute = routes[routes.length - 1]; // pick the most different alternative
        }
    }

    const optImpact = opt === 'recommended' ? null : {
        type: 'info',
        icon: cfg.icon,
        iconColor: cfg.color,
        title: cfg.title,
        desc: cfg.desc,
        result: cfg.result,
    };

    return { simRoute, optImpact };
}

// Open-Meteo Weather
async function fetchWeather(coords, dateStr, timeStr) {
    const url = `${METEO_API}?latitude=${coords.lat}&longitude=${coords.lon}&hourly=temperature_2m,precipitation_probability,weathercode&timezone=auto`;
    try {
        const response = await fetch(url);
        const data = await response.json();
        const targetDateTime = `${dateStr}T${timeStr.split(':')[0]}:00`;
        const timeIndex = data.hourly.time.findIndex(t => t.startsWith(targetDateTime));
        if (timeIndex !== -1) {
            return {
                temp: data.hourly.temperature_2m[timeIndex],
                precipProb: data.hourly.precipitation_probability[timeIndex],
                code: data.hourly.weathercode[timeIndex]
            };
        }
        return null;
    } catch(e) {
        console.error("Weather fetch failed", e);
        return null;
    }
}

// ============================================================
// ML Backend Call
// ============================================================
async function callMLBackend(baseRoute, weather, mode, dateStr, timeStr) {
    const baseMins = baseRoute.duration / 60;
    const distanceKm = baseRoute.distance / 1000;
    const precipProb = weather ? weather.precipProb : 0;

    try {
        const response = await fetch(ML_BACKEND, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                base_duration: baseMins,
                distance: distanceKm,
                precip_prob: precipProb,
                date_str: dateStr,
                time_str: timeStr,
                mode: mode
            })
        });

        if (!response.ok) throw new Error("서버 응답 오류");
        const data = await response.json();
        if (data.status !== 'success') throw new Error(data.message);

        const predictedDelay = data.predicted_delay_mins;
        const predictedTotal = data.predicted_total_mins;

        return {
            impacts: data.impacts, // Server now returns full impact array
            predictedDuration: predictedTotal * 60,
            predictedDistance: baseRoute.distance,
            addedDurationSec: predictedDelay * 60,
            addedCost: 0
        };

    } catch (error) {
        console.error("ML Backend Error:", error);
        alert("예측 서버(localhost:8000)에 연결할 수 없습니다.\n파이썬 서버가 실행 중인지 확인하세요.");
        return {
            impacts: [{
                type: 'info', icon: 'fa-server', iconColor: 'icon-info',
                title: '예측 서버 연결 실패',
                desc: '예측 서버와 통신할 수 없어 기본 소요 시간만 표시합니다. 터미널에서 python server.py를 실행해주세요.',
                result: '-'
            }],
            predictedDuration: baseRoute.duration,
            predictedDistance: baseRoute.distance,
            addedDurationSec: 0,
            addedCost: 0
        };
    }
}

// ============================================================
// Render Results
// ============================================================
function renderResults(baseRoute, impactAnalysis, fare) {
    const baseMins = Math.round(baseRoute.duration / 60);
    const baseKm = (baseRoute.distance / 1000).toFixed(1);
    
    let tollText = "";
    if (fare && fare.toll > 0) {
        tollText = ` / 통행료 ${fare.toll.toLocaleString()}원`;
    }
    document.getElementById('base-values').textContent = `${baseMins}분 / ${baseKm}km${tollText}`;

    const predMins = Math.round(impactAnalysis.predictedDuration / 60);
    const hourMins = predMins >= 60 ? `${Math.floor(predMins/60)}시간 ${predMins%60}분` : `${predMins}분`;
    document.getElementById('predicted-time').textContent = hourMins;

    const diffMins = Math.round(impactAnalysis.addedDurationSec / 60);
    const diffEl = document.getElementById('predicted-diff');

    if (diffMins > 0) {
        diffEl.innerHTML = `<i class="fa-solid fa-arrow-trend-up"></i> +${diffMins}분 지연 예상`;
        diffEl.className = 'predicted-diff diff-bad';
    } else if (diffMins < 0) {
        diffEl.innerHTML = `<i class="fa-solid fa-arrow-trend-down"></i> ${Math.abs(diffMins)}분 단축 예상`;
        diffEl.className = 'predicted-diff diff-good';
    } else {
        diffEl.innerHTML = `<i class="fa-solid fa-minus"></i> 지연 없음`;
        diffEl.className = 'predicted-diff diff-neutral';
    }

    impactList.innerHTML = '';
    impactAnalysis.impacts.forEach(impact => {
        const item = document.createElement('div');
        item.className = 'impact-item';
        item.innerHTML = `
            <div class="impact-icon ${impact.iconColor}">
                <i class="fa-solid ${impact.icon}"></i>
            </div>
            <div class="impact-content">
                <h4>${impact.title}</h4>
                <p>${impact.desc}</p>
                <span class="impact-result ${impact.type === 'info' ? 'text-blue' : 'text-red'}">${impact.result}</span>
            </div>
        `;
        impactList.appendChild(item);
    });

    resultsPanel.classList.remove('hidden');
}

// ============================================================
// Draw Route with Kakao Congestion Color Segments
// Kakao traffic_state: 0=Unknown, 1=Jam(Red), 2=Delay(Yellow), 3=Slow(Yellow-Green), 4=Good(Green)
// ============================================================
function drawKakaoCongestionRoute(sections, delayRatio) {
    if (routeLayer) {
        if (Array.isArray(routeLayer)) {
            routeLayer.forEach(l => map.removeLayer(l));
        } else {
            map.removeLayer(routeLayer);
        }
    }
    routeLayer = [];
    const segmentBounds = [];

    // Probability to downgrade a traffic state based on ML delay
    // If ML predicts huge delay, we shift colors towards Red.
    const downgradeProb = Math.min(0.9, delayRatio * 1.5);

    sections.forEach(section => {
        section.roads.forEach(road => {
            let state = road.traffic_state || 4; 
            
            // Apply ML penalty: randomly downgrade traffic state based on predicted future delay
            if (Math.random() < downgradeProb) {
                state = Math.max(1, state - 1); 
                if (Math.random() < downgradeProb * 0.5) state = Math.max(1, state - 1);
            }

            let color = '#3b82f6';
            let label = '구간';
            if (state === 1) { color = '#ef4444'; label = '🔴 정체 구간'; }
            else if (state === 2) { color = '#f59e0b'; label = '🟡 서행 구간'; }
            else if (state === 3 || state === 4) { color = '#22c55e'; label = '🟢 원활 구간'; }
            else { color = '#888888'; label = '⚪ 정보 없음'; }

            // Kakao vertexes are a flat array: [lon, lat, lon, lat, ...]
            const v = road.vertexes;
            const latlngs = [];
            for (let i = 0; i < v.length; i += 2) {
                latlngs.push([v[i+1], v[i]]); // [lat, lon]
            }

            const polyline = L.polyline(latlngs, {
                color,
                weight: state === 1 ? 6 : 5,
                opacity: 0.85,
                lineJoin: 'round',
                lineCap: 'round',
            }).addTo(map);

            polyline.bindTooltip(label, { sticky: true, className: 'congestion-tooltip' });
            
            routeLayer.push(polyline);
            segmentBounds.push(...latlngs);
        });
    });

    // Fit map to route
    if (segmentBounds.length > 0) {
        map.fitBounds(L.latLngBounds(segmentBounds), { padding: [50, 50] });
    }

    // Draw congestion legend on map
    drawCongestionLegend(delayRatio);
}

// Congestion Legend
let legendControl = null;
function drawCongestionLegend(delayRatio) {
    if (legendControl) map.removeControl(legendControl);

    const level = delayRatio > 0.3 ? '심각' : delayRatio > 0.15 ? '보통' : '원활';
    const levelColor = delayRatio > 0.3 ? '#ef4444' : delayRatio > 0.15 ? '#f59e0b' : '#22c55e';

    legendControl = L.control({ position: 'bottomleft' });
    legendControl.onAdd = function() {
        const div = L.DomUtil.create('div', 'congestion-legend');
        div.innerHTML = `
            <div class="legend-title">경로 혼잡도 예측</div>
            <div class="legend-row"><span class="legend-dot" style="background:#22c55e"></span> 원활</div>
            <div class="legend-row"><span class="legend-dot" style="background:#f59e0b"></span> 서행</div>
            <div class="legend-row"><span class="legend-dot" style="background:#ef4444"></span> 정체</div>
            <div class="legend-overall" style="color:${levelColor}">전체: ${level}</div>
        `;
        return div;
    };
    legendControl.addTo(map);
}

// Start
window.onload = init;
