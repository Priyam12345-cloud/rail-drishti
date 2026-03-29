# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC ## Indian Railways Fare Calculator
# MAGIC Type any city name (Delhi, Mumbai, Lucknow...) → Get trains with fares for **all 6 classes**: GEN | SL | CC | 3AC | 2AC | 1AC
# MAGIC
# MAGIC **Data**: Datameet Indian Railways — 5,208 trains · 8,990 stations · 417K schedule stops

# COMMAND ----------

# DBTITLE 1,Load Railway Data
import json
import math
from collections import defaultdict
from pyspark.sql import functions as F

DATA_PATH = "/Workspace/Users/lopamudra.wncc@gmail.com/Data"

# ── 1. Schedules (417K stops) ──
with open(f"{DATA_PATH}/schedules.json", "r") as f:
    schedules_raw = json.load(f)

# ── 2. Stations (8,990 with coordinates) ──
with open(f"{DATA_PATH}/stations.json", "r") as f:
    stations_geo = json.load(f)

station_lookup = {}   # code -> {name, state, lat, lng}
for feat in stations_geo["features"]:
    p = feat["properties"]
    geom = feat.get("geometry")
    coords = geom["coordinates"] if geom and geom.get("coordinates") else [None, None]
    station_lookup[p["code"]] = {
        "name": p.get("name", p["code"]),
        "state": p.get("state", ""),
        "lat": coords[1],
        "lng": coords[0],
    }

# ── 3. Trains (5,208 with class availability & type) ──
with open(f"{DATA_PATH}/trains.json", "r") as f:
    trains_geo = json.load(f)

train_info = {}  # train_number -> {name, type, classes, from, to}
for feat in trains_geo["features"]:
    p = feat["properties"]
    tn = str(p.get("number", ""))
    available_classes = []
    if int(p.get("sleeper", 0) or 0): available_classes.append("SL")
    if int(p.get("third_ac", 0) or 0): available_classes.append("3AC")
    if int(p.get("second_ac", 0) or 0): available_classes.append("2AC")
    if int(p.get("first_ac", 0) or 0): available_classes.append("1AC")
    if int(p.get("first_class", 0) or 0): available_classes.append("FC")
    if int(p.get("chair_car", 0) or 0): available_classes.append("CC")
    # All trains have 2S (unreserved) by default for general coaches
    if "SL" in available_classes or len(available_classes) == 0:
        available_classes.insert(0, "2S")
    train_info[tn] = {
        "name": p.get("name", ""),
        "type": p.get("type", "Mail/Exp"),
        "from_code": p.get("from_station_code", ""),
        "from_name": p.get("from_station_name", ""),
        "to_code": p.get("to_station_code", ""),
        "to_name": p.get("to_station_name", ""),
        "classes": available_classes,
        "departure": p.get("departure", ""),
        "arrival": p.get("arrival", ""),
        "duration_h": p.get("duration_h", 0),
        "duration_m": p.get("duration_m", 0),
    }

# ── 4. Build schedule index: train -> ordered stops ──
train_stops = defaultdict(list)  # train_number -> sorted list of stops
for s in schedules_raw:
    train_stops[str(s["train_number"])].append(s)

for tn in train_stops:
    train_stops[tn].sort(key=lambda x: (x.get("day", 1) or 1, x.get("id", 0) or 0))

print(f"Loaded: {len(station_lookup):,} stations | {len(train_info):,} trains | {len(schedules_raw):,} schedule stops")
print(f"Sample station: NDLS -> {station_lookup.get('NDLS', 'not found')}")
print(f"Sample train: 12301 -> {train_info.get('12301', {}).get('name', 'not found')}")
print(f"Sample stops for 12301: {len(train_stops.get('12301', []))} stops")

# COMMAND ----------

# DBTITLE 1,Fare Calculation Engine
# ============================================================
# FARE CALCULATION ENGINE + CONNECTING TRAINS
# Indian Railways formula-based pricing (July 2025 revision)
# Shows ALL class fares: GEN, SL, CC, 3AC, 2AC, 1AC
# Smart station search + via-hub connecting train routes
# ============================================================

ALL_CLASSES = ["GEN", "SL", "CC", "3AC", "2AC", "1AC"]

CITY_ALIASES = {
    "DELHI": "NDLS", "NEW DELHI": "NDLS", "NEWDELHI": "NDLS",
    "MUMBAI": "BCT", "BOMBAY": "BCT", "MUMBAI CENTRAL": "BCT",
    "KOLKATA": "HWH", "CALCUTTA": "HWH", "HOWRAH": "HWH",
    "CHENNAI": "MAS", "MADRAS": "MAS", "CHENNAI CENTRAL": "MAS",
    "BANGALORE": "SBC", "BENGALURU": "SBC", "BANGLORE": "SBC",
    "HYDERABAD": "SC", "SECUNDERABAD": "SC",
    "PUNE": "PUNE", "POONA": "PUNE",
    "JAIPUR": "JP", "JAYPUR": "JP",
    "LUCKNOW": "LKO", "LAKHNAU": "LKO",
    "AHMEDABAD": "ADI", "AHEMDABAD": "ADI", "AMADAVAD": "ADI",
    "PATNA": "PNBE", "PATNA JN": "PNBE",
    "BHOPAL": "BPL", "BHOPAL JN": "BPL",
    "KANPUR": "CNB", "KANPUR CENTRAL": "CNB",
    "AGRA": "AGC", "AGRA CANTT": "AGC",
    "VARANASI": "BSB", "BANARAS": "BSB", "KASHI": "BSB",
    "NAGPUR": "NGP", "NAGPUR JN": "NGP",
    "CHANDIGARH": "CDG",
    "INDORE": "INDB", "INDORE JN": "INDB",
    "COIMBATORE": "CBE", "KOVAI": "CBE",
    "THIRUVANANTHAPURAM": "TVC", "TRIVANDRUM": "TVC",
    "KOCHI": "ERS", "ERNAKULAM": "ERS", "COCHIN": "ERS",
    "VISAKHAPATNAM": "VSKP", "VIZAG": "VSKP",
    "GUWAHATI": "GHY",
    "RANCHI": "RNC", "RANCHI JN": "RNC",
    "DEHRADUN": "DDN",
    "AMRITSAR": "ASR",
    "JAMMU": "JAT", "JAMMU TAWI": "JAT",
    "GOA": "MAO", "MADGAON": "MAO", "MARGAO": "MAO",
    "SURAT": "ST",
    "VADODARA": "BRC", "BARODA": "BRC",
    "JODHPUR": "JU",
    "UDAIPUR": "UDZ",
    "ALLAHABAD": "ALD", "PRAYAGRAJ": "PRYJ",
    "GWALIOR": "GWL",
    "MYSORE": "MYS", "MYSURU": "MYS",
    "TIRUPATI": "TPTY",
    "UJJAIN": "UJN",
    "RAIPUR": "R",
    "GORAKHPUR": "GKP",
}

def haversine_km(lat1, lng1, lat2, lng2):
    if any(v is None for v in (lat1, lng1, lat2, lng2)):
        return None
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def route_distance_km(train_number, from_code, to_code):
    stops = train_stops.get(train_number, [])
    if not stops:
        return None
    from_idx, to_idx = None, None
    for i, s in enumerate(stops):
        code = s.get("station_code", "")
        if code == from_code and from_idx is None:
            from_idx = i
        if code == to_code and from_idx is not None:
            to_idx = i
    if from_idx is None or to_idx is None or from_idx >= to_idx:
        s1 = station_lookup.get(from_code, {})
        s2 = station_lookup.get(to_code, {})
        d = haversine_km(s1.get("lat"), s1.get("lng"), s2.get("lat"), s2.get("lng"))
        return round(d * 1.3) if d else None
    total = 0.0
    for i in range(from_idx, to_idx):
        c1 = stops[i].get("station_code", "")
        c2 = stops[i+1].get("station_code", "")
        s1 = station_lookup.get(c1, {})
        s2 = station_lookup.get(c2, {})
        seg = haversine_km(s1.get("lat"), s1.get("lng"), s2.get("lat"), s2.get("lng"))
        if seg:
            total += seg
    return round(total * 1.2) if total > 0 else None

# Fare tables
PER_KM_RATES = {"GEN": 0.22, "SL": 0.45, "CC": 1.10, "3AC": 1.20, "2AC": 1.85, "1AC": 3.10}
RSV_CHARGE = {"GEN": 0, "SL": 40, "CC": 40, "3AC": 40, "2AC": 50, "1AC": 60}
SF_SURCHARGE = {"GEN": 0, "SL": 30, "CC": 30, "3AC": 45, "2AC": 50, "1AC": 75}
TYPE_MULTIPLIER = {
    "Rajdhani": 1.35, "Shatabdi": 1.30, "Duronto": 1.25,
    "Vande Bharat": 1.40, "Garib Rath": 0.75, "Tejas": 1.45, "Humsafar": 1.15,
}

def calculate_fare(distance_km, travel_class, train_type="", train_number=""):
    if distance_km is None or distance_km <= 0:
        return None
    rate = PER_KM_RATES.get(travel_class, 0.45)
    base = round(distance_km * rate)
    for keyword, mult in TYPE_MULTIPLIER.items():
        if keyword.lower() in (train_type or "").lower():
            base = round(base * mult)
            break
    rsv = RSV_CHARGE.get(travel_class, 40)
    sf = 0
    if train_number and (train_number.startswith("12") or train_number.startswith("22")):
        sf = SF_SURCHARGE.get(travel_class, 30)
    return {"base": base, "rsv": rsv, "sf": sf, "total": base + rsv + sf}

def _get_train_result(train_no, from_code, to_code, stops, fi, ti):
    """Build a result dict for a single train leg."""
    dep_stop = stops[fi]
    arr_stop = stops[ti]
    info = train_info.get(train_no, {})
    dist = route_distance_km(train_no, from_code, to_code)
    fares = {}
    for cls in ALL_CLASSES:
        f = calculate_fare(dist, cls, info.get("type", ""), train_no)
        if f:
            fares[cls] = f
    dep_time = dep_stop.get("departure", "") or dep_stop.get("arrival", "")
    arr_time = arr_stop.get("arrival", "") or arr_stop.get("departure", "")
    return {
        "train_no": train_no,
        "train_name": info.get("name", train_no),
        "train_type": info.get("type", ""),
        "from": station_lookup.get(from_code, {}).get("name", from_code),
        "to": station_lookup.get(to_code, {}).get("name", to_code),
        "from_code": from_code,
        "to_code": to_code,
        "departure": dep_time,
        "arrival": arr_time,
        "dep_day": dep_stop.get("day") or 1,
        "arr_day": arr_stop.get("day") or 1,
        "distance_km": dist,
        "stops": ti - fi,
        "classes": ALL_CLASSES,
        "fares": fares,
    }

def find_trains_between(from_code, to_code):
    """Find all DIRECT trains between two stations."""
    results = []
    seen_trains = set()
    for train_no, stops in train_stops.items():
        codes = [s.get("station_code", "") for s in stops]
        if from_code in codes and to_code in codes:
            fi = codes.index(from_code)
            ti = codes.index(to_code)
            if fi < ti and train_no not in seen_trains:
                seen_trains.add(train_no)
                results.append(_get_train_result(train_no, from_code, to_code, stops, fi, ti))
    def sort_key(r):
        t = (r["departure"] or "").replace("None", "99:99")
        try:
            parts = t.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return 9999
    results.sort(key=sort_key)
    return results

# ============================================================
# CONNECTING TRAINS ENGINE
# Finds 2-leg routes: A -> Hub -> B via major junctions
# ============================================================

print("Building station-trains index for connecting routes...")
station_trains_index = defaultdict(set)
for train_no, stops in train_stops.items():
    for s in stops:
        code = s.get("station_code", "")
        if code:
            station_trains_index[code].add(train_no)

HUB_MIN_TRAINS = 15
hub_stations = {code for code, trains in station_trains_index.items() if len(trains) >= HUB_MIN_TRAINS}
print(f"Hub stations (>={HUB_MIN_TRAINS} trains): {len(hub_stations)}")

def _time_to_mins(t, day=1):
    """Convert time string + day to minutes. None-safe."""
    try:
        day = int(day or 1)
        t = str(t or "").strip()
        parts = t.replace(".", ":").split(":")
        h = int(parts[0]) if parts[0] not in ('None', '', 'null') else 0
        m = int(parts[1]) if len(parts) > 1 and parts[1] not in ('None', '', 'null') else 0
        return (day - 1) * 1440 + h * 60 + m
    except:
        return (int(day or 1) - 1) * 1440

def _format_dur(mins):
    if mins <= 0: mins += 1440
    h, m = divmod(mins, 60)
    return str(h) + "h " + str(m) + "m"

def _find_leg(from_code, to_code, max_results=3):
    """Find best trains for a single leg, sorted by departure."""
    results = []
    seen = set()
    for tn in station_trains_index.get(from_code, set()) & station_trains_index.get(to_code, set()):
        if tn in seen:
            continue
        stops = train_stops.get(tn, [])
        codes = [s.get("station_code", "") for s in stops]
        if from_code in codes and to_code in codes:
            fi = codes.index(from_code)
            ti = codes.index(to_code)
            if fi < ti:
                seen.add(tn)
                results.append(_get_train_result(tn, from_code, to_code, stops, fi, ti))
    results.sort(key=lambda r: _time_to_mins(r["departure"], r.get("dep_day") or 1))
    return results[:max_results]

def find_connecting_trains(from_code, to_code, max_hubs=12, max_connections=8):
    """Find 2-leg connecting routes: from_code -> Hub -> to_code."""
    origin_trains = station_trains_index.get(from_code, set())
    dest_trains = station_trains_index.get(to_code, set())
    
    candidate_hubs = []
    for hub in hub_stations:
        if hub == from_code or hub == to_code:
            continue
        hub_trains = station_trains_index.get(hub, set())
        from_hub = origin_trains & hub_trains
        hub_to = hub_trains & dest_trains
        if from_hub and hub_to:
            candidate_hubs.append((hub, len(from_hub) + len(hub_to)))
    
    candidate_hubs.sort(key=lambda x: -x[1])
    candidate_hubs = candidate_hubs[:max_hubs]
    
    connections = []
    seen_combos = set()
    
    for hub_code, _ in candidate_hubs:
        hub_name = station_lookup.get(hub_code, {}).get("name", hub_code)
        leg1_options = _find_leg(from_code, hub_code, max_results=3)
        leg2_options = _find_leg(hub_code, to_code, max_results=3)
        if not leg1_options or not leg2_options:
            continue
        
        MIN_CONN_MINS = 30
        for l1 in leg1_options:
            l1_arr = _time_to_mins(l1["arrival"], l1.get("arr_day") or 1)
            for l2 in leg2_options:
                combo_key = l1["train_no"] + "_" + l2["train_no"] + "_" + hub_code
                if combo_key in seen_combos:
                    continue
                l2_dep_tod = _time_to_mins(l2["departure"])
                l2_dep = l2_dep_tod
                while l2_dep < l1_arr + MIN_CONN_MINS:
                    l2_dep += 1440
                conn_wait = l2_dep - l1_arr
                if conn_wait > 1440 * 2:
                    continue
                l2_dur = _time_to_mins(l2["arrival"], l2.get("arr_day") or 1) - _time_to_mins(l2["departure"], l2.get("dep_day") or 1)
                if l2_dur <= 0:
                    l2_dur += 1440
                l2_arr = l2_dep + l2_dur
                l1_dep = _time_to_mins(l1["departure"], l1.get("dep_day") or 1)
                total_mins = l2_arr - l1_dep
                if total_mins <= 0:
                    total_mins += 1440
                if total_mins > 1440 * 4:
                    continue
                seen_combos.add(combo_key)
                combined_fares = {}
                for cls in ALL_CLASSES:
                    f1 = l1["fares"].get(cls)
                    f2 = l2["fares"].get(cls)
                    if f1 and f2:
                        combined_fares[cls] = {
                            "total": f1["total"] + f2["total"],
                            "leg1": f1["total"],
                            "leg2": f2["total"],
                        }
                connections.append({
                    "type": "connecting",
                    "hub_code": hub_code,
                    "hub_name": hub_name,
                    "leg1": l1,
                    "leg2": l2,
                    "conn_wait_mins": conn_wait,
                    "conn_wait": _format_dur(conn_wait),
                    "total_mins": total_mins,
                    "total_time": _format_dur(total_mins),
                    "combined_fares": combined_fares,
                    "total_distance": (l1.get("distance_km") or 0) + (l2.get("distance_km") or 0),
                })
    
    connections.sort(key=lambda c: c["total_mins"])
    return connections[:max_connections]

def search_station(query):
    """Smart station search: handles city names, abbreviations, partial matches."""
    query = query.strip().upper()
    if not query:
        return []
    if query in CITY_ALIASES:
        code = CITY_ALIASES[query]
        info = station_lookup.get(code, {})
        if info:
            return [(code, info["name"], info.get("state", ""))]
    if query in station_lookup:
        info = station_lookup[query]
        return [(query, info["name"], info.get("state", ""))]
    exact = []
    for code, info in station_lookup.items():
        name = (info["name"] or "").upper()
        if name == query:
            exact.append((code, info["name"], info.get("state", "")))
    if exact:
        return exact[:10]
    for alias, code in CITY_ALIASES.items():
        if query in alias or alias in query:
            info = station_lookup.get(code, {})
            if info:
                return [(code, info["name"], info.get("state", ""))]
    query_words = query.split()
    scored = []
    for code, info in station_lookup.items():
        name = (info["name"] or "").upper()
        if query in name:
            scored.append((2, code, info["name"], info.get("state", "")))
        elif query in code:
            scored.append((1, code, info["name"], info.get("state", "")))
        elif any(w in name for w in query_words if len(w) >= 3):
            scored.append((0, code, info["name"], info.get("state", "")))
    scored.sort(key=lambda x: (-x[0], x[2]))
    return [(code, name, state) for _, code, name, state in scored[:15]]

print("Fare Engine ready: direct + connecting trains, all 6 classes!")
print(f"   Classes: {' | '.join(ALL_CLASSES)}")
print(f"\nTest connecting: Gorakhpur -> Goa")
conns = find_connecting_trains("GKP", "MAO", max_hubs=8, max_connections=3)
print(f"   Found {len(conns)} connecting routes")
for c in conns[:3]:
    print(f"   via {c['hub_name']} ({c['hub_code']}): {c['total_time']} | wait {c['conn_wait']}")
    sl = c['combined_fares'].get('SL', {})
    if sl:
        print(f"     SL fare: {chr(8377)}{sl['total']} ({chr(8377)}{sl['leg1']} + {chr(8377)}{sl['leg2']})")


# COMMAND ----------

# DBTITLE 1,Interactive Fare Calculator
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from datetime import date, datetime, timedelta

CSS = """
<style>
.fare-container { font-family: 'Segoe UI', Arial, sans-serif; max-width: 960px; }
.search-header { background: linear-gradient(135deg, #1a237e, #283593); color: white;
    padding: 20px 25px; border-radius: 12px 12px 0 0; }
.search-header h2 { margin: 0 0 5px 0; font-size: 20px; }
.search-header p { margin: 0; opacity: 0.8; font-size: 13px; }
.route-banner { background: #e8eaf6; padding: 14px 25px; border-bottom: 2px solid #3949ab;
    display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
.route-banner .station { font-size: 16px; font-weight: 700; color: #1a237e; }
.route-banner .arrow { font-size: 22px; color: #5c6bc0; }
.route-banner .travel-date { margin-left: auto; background: #3949ab; color: white;
    padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 600; }
.train-card { border: 1px solid #e0e0e0; margin: 8px 0; border-radius: 8px;
    overflow: hidden; transition: box-shadow 0.2s; }
.train-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.12); }
.train-header { padding: 12px 18px; display: flex; align-items: center;
    background: #fafafa; border-bottom: 1px solid #eee; }
.train-no { background: #1a237e; color: white; padding: 3px 10px; border-radius: 4px;
    font-size: 12px; font-weight: 700; margin-right: 12px; }
.train-name { font-weight: 600; font-size: 14px; color: #333; flex: 1; }
.train-type-badge { background: #e8eaf6; color: #3949ab; padding: 2px 8px;
    border-radius: 10px; font-size: 11px; font-weight: 600; }
.train-body { padding: 14px 18px; }
.time-row { display: flex; align-items: center; margin-bottom: 12px; }
.time-block { text-align: center; min-width: 90px; }
.time-block .time { font-size: 18px; font-weight: 700; color: #1a237e; }
.time-block .label { font-size: 11px; color: #999; }
.time-block .day { font-size: 10px; color: #e65100; font-weight: 600; }
.duration-line { flex: 1; text-align: center; position: relative; margin: 0 15px; }
.duration-line .dur { font-size: 11px; color: #666; }
.duration-line .dist { font-size: 10px; color: #999; }
.duration-line hr { border: none; border-top: 2px dashed #ccc; margin: 4px 0; }
.fare-row { display: flex; gap: 6px; flex-wrap: wrap; }
.fare-chip { border: 1px solid #e0e0e0; border-radius: 6px; padding: 6px 10px;
    text-align: center; min-width: 80px; background: white; }
.fare-chip .cls { font-size: 11px; font-weight: 700; color: #5c6bc0; }
.fare-chip .price { font-size: 15px; font-weight: 700; color: #1b5e20; }
.fare-chip .detail { font-size: 9px; color: #999; }
.fare-chip.cheapest { border-color: #4caf50; background: #e8f5e9; }
.no-trains { padding: 40px; text-align: center; color: #999; }
.station-suggest { background: #fff3e0; padding: 8px 15px; border-radius: 6px;
    margin: 4px 0; font-size: 12px; }
.summary-bar { padding: 10px 25px; background: #f5f5f5; border-bottom: 1px solid #e0e0e0;
    font-size: 13px; color: #666; }
.section-title { padding: 14px 25px; background: #fff3e0; border-bottom: 2px solid #e65100;
    font-size: 15px; font-weight: 700; color: #bf360c; }
.conn-card { border: 1px solid #ffe0b2; margin: 8px 0; border-radius: 8px;
    overflow: hidden; transition: box-shadow 0.2s; }
.conn-card:hover { box-shadow: 0 2px 12px rgba(230,81,0,0.15); }
.conn-header { padding: 10px 18px; background: #fff8e1; border-bottom: 1px solid #ffe0b2;
    display: flex; align-items: center; gap: 10px; }
.conn-via-badge { background: #e65100; color: white; padding: 3px 10px; border-radius: 4px;
    font-size: 11px; font-weight: 700; }
.conn-total { margin-left: auto; font-size: 13px; color: #bf360c; font-weight: 600; }
.conn-body { padding: 12px 18px; }
.leg-section { margin-bottom: 10px; padding: 10px; background: #fafafa; border-radius: 6px;
    border-left: 3px solid #3949ab; }
.leg-label { font-size: 10px; font-weight: 700; color: #5c6bc0; margin-bottom: 4px; }
.leg-train { font-size: 13px; font-weight: 600; color: #333; }
.leg-details { font-size: 12px; color: #666; margin-top: 2px; }
.hub-divider { text-align: center; padding: 6px; font-size: 12px; color: #e65100;
    font-weight: 600; background: #fff3e0; border-radius: 4px; margin: 4px 0; }
</style>
"""

CLASS_LABELS = {"GEN": "General", "SL": "Sleeper", "CC": "Chair Car",
                "3AC": "3rd AC", "2AC": "2nd AC", "1AC": "1st AC"}
CAL_ICON = "&#x1F4C5;"
RUPEE = chr(8377)
DOT = chr(183)
ARROW = chr(10140)

def format_time(t):
    if not t or t == "None": return "--:--"
    return t[:5] if len(t) >= 5 else t

def compute_duration(dep, arr, dep_day, arr_day):
    try:
        dp = dep.split(":")
        ap = arr.split(":")
        dep_mins = int(dp[0]) * 60 + int(dp[1])
        arr_mins = int(ap[0]) * 60 + int(ap[1])
        total = arr_mins - dep_mins + (int(arr_day or 1) - int(dep_day or 1)) * 1440
        if total <= 0: total += 1440
        h, m = divmod(total, 60)
        return str(h) + "h " + str(m) + "m"
    except:
        return "--"

def get_calendar_dates(travel_date, dep_day, arr_day):
    try:
        extra_days = (int(arr_day or 1) - int(dep_day or 1))
        dep_str = travel_date.strftime("%a, %b %d")
        arr_date = travel_date + timedelta(days=extra_days)
        arr_str = arr_date.strftime("%a, %b %d")
        return dep_str, arr_str
    except:
        return "", ""

def render_train_card(train, travel_date):
    dep = format_time(train["departure"])
    arr = format_time(train["arrival"])
    dur = compute_duration(train["departure"], train["arrival"], train["dep_day"], train["arr_day"])
    dist_text = str(train['distance_km']) + " km" if train['distance_km'] else "N/A"
    dep_date_str, arr_date_str = get_calendar_dates(travel_date, train["dep_day"], train["arr_day"])
    min_fare = min((f["total"] for f in train["fares"].values()), default=0)
    fare_chips = ""
    for cls in ALL_CLASSES:
        if cls in train["fares"]:
            f = train["fares"][cls]
            is_cheap = "cheapest" if f["total"] == min_fare else ""
            detail = "Base " + RUPEE + str(f['base'])
            if f['sf'] > 0: detail += " +SF " + RUPEE + str(f['sf'])
            if f['rsv'] > 0: detail += " +Rsv " + RUPEE + str(f['rsv'])
            label = CLASS_LABELS.get(cls, cls)
            fare_chips += (
                '<div class="fare-chip ' + is_cheap + '">'
                '<div class="cls">' + cls + ' <span style="font-weight:400;font-size:9px;">(' + label + ')</span></div>'
                '<div class="price">' + RUPEE + f"{f['total']:,}" + '</div>'
                '<div class="detail">' + detail + '</div></div>'
            )
    return (
        '<div class="train-card"><div class="train-header">'
        '<span class="train-no">' + train["train_no"] + '</span>'
        '<span class="train-name">' + train["train_name"] + '</span>'
        '<span class="train-type-badge">' + (train["train_type"] or "Express") + '</span>'
        '</div><div class="train-body"><div class="time-row">'
        '<div class="time-block"><div class="time">' + dep + '</div>'
        '<div class="label">' + train["from"] + '</div>'
        '<div class="day">' + dep_date_str + '</div></div>'
        '<div class="duration-line"><div class="dur">' + dur + ' ' + DOT + ' ' + str(train["stops"]) + ' stops</div>'
        '<hr><div class="dist">' + dist_text + '</div></div>'
        '<div class="time-block"><div class="time">' + arr + '</div>'
        '<div class="label">' + train["to"] + '</div>'
        '<div class="day">' + arr_date_str + '</div></div>'
        '</div><div class="fare-row">' + fare_chips + '</div></div></div>'
    )

def render_connecting_card(conn, travel_date):
    """Render a 2-leg connecting train card."""
    l1 = conn["leg1"]
    l2 = conn["leg2"]
    hub = conn["hub_name"] + " (" + conn["hub_code"] + ")"
    
    l1_dep = format_time(l1["departure"])
    l1_arr = format_time(l1["arrival"])
    l2_dep = format_time(l2["departure"])
    l2_arr = format_time(l2["arrival"])
    l1_dur = compute_duration(l1["departure"], l1["arrival"], l1["dep_day"], l1["arr_day"])
    l2_dur = compute_duration(l2["departure"], l2["arrival"], l2["dep_day"], l2["arr_day"])
    dep_date_str, _ = get_calendar_dates(travel_date, l1["dep_day"], l1["arr_day"])
    dist_total = str(conn.get("total_distance", 0)) + " km" if conn.get("total_distance") else ""
    
    # Combined fare chips
    fare_chips = ""
    cf = conn.get("combined_fares", {})
    min_total = min((f["total"] for f in cf.values()), default=0) if cf else 0
    for cls in ALL_CLASSES:
        if cls in cf:
            f = cf[cls]
            is_cheap = "cheapest" if f["total"] == min_total else ""
            detail = RUPEE + str(f["leg1"]) + " + " + RUPEE + str(f["leg2"])
            label = CLASS_LABELS.get(cls, cls)
            fare_chips += (
                '<div class="fare-chip ' + is_cheap + '">'
                '<div class="cls">' + cls + ' <span style="font-weight:400;font-size:9px;">(' + label + ')</span></div>'
                '<div class="price">' + RUPEE + f"{f['total']:,}" + '</div>'
                '<div class="detail">' + detail + '</div></div>'
            )
    
    l1_info = train_info.get(l1["train_no"], {})
    l2_info = train_info.get(l2["train_no"], {})
    
    return (
        '<div class="conn-card"><div class="conn-header">'
        '<span class="conn-via-badge">VIA ' + conn["hub_code"] + '</span>'
        '<span style="font-size:13px;color:#333;font-weight:600;">'
        + l1["from"] + ' ' + ARROW + ' ' + hub + ' ' + ARROW + ' ' + l2["to"] + '</span>'
        '<span class="conn-total">' + conn["total_time"] + ' ' + DOT + ' ' + dist_total + '</span>'
        '</div><div class="conn-body">'
        # Leg 1
        '<div class="leg-section">'
        '<div class="leg-label">LEG 1: ' + l1["from"] + ' ' + ARROW + ' ' + conn["hub_name"] + '</div>'
        '<div class="leg-train">' + l1["train_no"] + ' ' + l1["train_name"] + '</div>'
        '<div class="leg-details">' + l1_dep + ' ' + ARROW + ' ' + l1_arr + ' ' + DOT + ' ' + l1_dur
        + ' ' + DOT + ' ' + str(l1.get("distance_km", "")) + ' km</div></div>'
        # Hub divider
        '<div class="hub-divider">Change at ' + hub + ' ' + DOT + ' Wait: ' + conn["conn_wait"] + '</div>'
        # Leg 2
        '<div class="leg-section">'
        '<div class="leg-label">LEG 2: ' + conn["hub_name"] + ' ' + ARROW + ' ' + l2["to"] + '</div>'
        '<div class="leg-train">' + l2["train_no"] + ' ' + l2["train_name"] + '</div>'
        '<div class="leg-details">' + l2_dep + ' ' + ARROW + ' ' + l2_arr + ' ' + DOT + ' ' + l2_dur
        + ' ' + DOT + ' ' + str(l2.get("distance_km", "")) + ' km</div></div>'
        # Combined fares
        '<div style="margin-top:10px;"><div style="font-size:11px;color:#888;margin-bottom:4px;">Combined fare (Leg 1 + Leg 2):</div>'
        '<div class="fare-row">' + fare_chips + '</div></div>'
        '</div></div>'
    )

def on_search(btn):
    with output:
        clear_output()
        from_q = from_input.value.strip()
        to_q = to_input.value.strip()
        travel_date = date_picker.value
        if not from_q or not to_q:
            display(HTML('<div class="no-trains"><p>Enter both stations</p></div>')); return
        from_matches = search_station(from_q)
        to_matches = search_station(to_q)
        html = CSS
        if not from_matches:
            html += '<div class="station-suggest">No station found for "' + from_q + '"</div>'
            display(HTML(html)); return
        if not to_matches:
            html += '<div class="station-suggest">No station found for "' + to_q + '"</div>'
            display(HTML(html)); return
        from_code, from_name = from_matches[0][0], from_matches[0][1]
        to_code, to_name = to_matches[0][0], to_matches[0][1]
        if len(from_matches) > 1 and from_q.strip().upper() not in (from_code, from_name.upper()) and from_q.strip().upper() not in CITY_ALIASES:
            s = ", ".join([c + " (" + n + ")" for c, n, _ in from_matches[:5]])
            html += '<div class="station-suggest">Matched "' + from_q + '" to <b>' + from_code + ' (' + from_name + ')</b>. Others: ' + s + '</div>'
        if len(to_matches) > 1 and to_q.strip().upper() not in (to_code, to_name.upper()) and to_q.strip().upper() not in CITY_ALIASES:
            s = ", ".join([c + " (" + n + ")" for c, n, _ in to_matches[:5]])
            html += '<div class="station-suggest">Matched "' + to_q + '" to <b>' + to_code + ' (' + to_name + ')</b>. Others: ' + s + '</div>'
        
        # Search both direct and connecting trains
        direct_trains = find_trains_between(from_code, to_code)
        connecting = find_connecting_trains(from_code, to_code)
        
        date_display = travel_date.strftime("%A, %B %d, %Y")
        date_short = travel_date.strftime("%a, %b %d")
        
        html += '<div class="fare-container">'
        html += (
            '<div class="search-header">'
            '<h2>Train Fare Calculator</h2>'
            '<p>Indian Railways | Direct + Connecting trains | All 6 classes</p>'
            '</div>'
            '<div class="route-banner">'
            '<span class="station">' + from_code + ' ' + DOT + ' ' + from_name + '</span>'
            '<span class="arrow">' + ARROW + '</span>'
            '<span class="station">' + to_code + ' ' + DOT + ' ' + to_name + '</span>'
            '<span class="travel-date">' + CAL_ICON + ' ' + date_display + '</span>'
            '</div>'
        )
        
        total_found = len(direct_trains) + len(connecting)
        
        if total_found == 0:
            html += '<div class="no-trains"><p><b>No trains found</b> (direct or connecting)</p></div>'
        else:
            # Direct trains section
            if direct_trains:
                dist_info = " | ~" + str(direct_trains[0]["distance_km"]) + " km" if direct_trains[0]["distance_km"] else ""
                html += (
                    '<div class="summary-bar">DIRECT: ' + str(len(direct_trains))
                    + ' train' + ('s' if len(direct_trains) != 1 else '')
                    + dist_info + ' | ' + date_short + ' | GEN | SL | CC | 3AC | 2AC | 1AC</div>'
                )
                for t in direct_trains:
                    html += render_train_card(t, travel_date)
            else:
                html += '<div class="summary-bar">No direct trains found on this route</div>'
            
            # Connecting trains section
            if connecting:
                html += (
                    '<div class="section-title">CONNECTING TRAINS: '
                    + str(len(connecting)) + ' route' + ('s' if len(connecting) != 1 else '')
                    + ' via intermediate stations</div>'
                )
                for c in connecting:
                    html += render_connecting_card(c, travel_date)
        
        html += '</div>'
        display(HTML(html))

style = {'description_width': '55px'}
layout_stn = widgets.Layout(width='280px')
layout_date = widgets.Layout(width='200px')
from_input = widgets.Text(value='Gorakhpur', placeholder='City: Delhi, Mumbai...',
                          description='From:', style=style, layout=layout_stn)
to_input = widgets.Text(value='Goa', placeholder='City: Lucknow, Jaipur...',
                        description='To:', style=style, layout=layout_stn)
date_picker = widgets.DatePicker(value=date.today(), description='Date:',
                                  style=style, layout=layout_date)
search_btn = widgets.Button(description='Search Trains & Fares', button_style='primary',
                            layout=widgets.Layout(width='280px', height='40px'))
output = widgets.Output()
search_btn.on_click(on_search)
swap_btn = widgets.Button(description=chr(8644), button_style='info',
                          layout=widgets.Layout(width='45px', height='30px'))
def on_swap(btn): from_input.value, to_input.value = to_input.value, from_input.value
swap_btn.on_click(on_swap)
header_html = widgets.HTML('<h3 style="color:#1a237e;margin:0 0 10px 0;">Search direct + connecting trains with fares</h3>')
row1 = widgets.HBox([from_input, swap_btn, to_input, date_picker],
                    layout=widgets.Layout(gap='8px', align_items='center'))
display(header_html, row1, search_btn, output)
on_search(None)

# COMMAND ----------

# DBTITLE 1,Quick Search: Popular Routes
# ============================================================
# QUICK SEARCH: Type any route and run this cell
# ============================================================

# ✏️ CHANGE THESE → then run the cell
FROM_STATION = "NDLS"   # Station code or name: NDLS, HWH, MAS, BCT, PUNE, JP, LKO, BLR, BPL
TO_STATION   = "LKO"    # Try: LUCKNOW, MUMBAI, HOWRAH, CHENNAI, JAIPUR, PUNE, BANGALORE

# ── Search & Display ──
from_matches = search_station(FROM_STATION)
to_matches = search_station(TO_STATION)

if from_matches and to_matches:
    fc, fn = from_matches[0][0], from_matches[0][1]
    tc, tn = to_matches[0][0], to_matches[0][1]
    print(f"\n🚉 {fn} ({fc}) ➜ {tn} ({tc})")
    print("=" * 70)
    
    trains = find_trains_between(fc, tc)
    print(f"\n📊 {len(trains)} trains found\n")
    
    # Build rows with ALL 6 class fares
    rows = []
    for t in trains:
        base_row = {
            "Train No": t["train_no"],
            "Train Name": t["train_name"],
            "Type": t["train_type"] or "Express",
            "Departure": t["departure"][:5] if t["departure"] else "--",
            "Arrival": t["arrival"][:5] if t["arrival"] else "--",
            "Distance (km)": t["distance_km"],
            "Stops": t["stops"],
        }
        # Add fare columns for ALL 6 classes
        for cls in ALL_CLASSES:  # GEN, SL, CC, 3AC, 2AC, 1AC
            if cls in t["fares"]:
                base_row[f"₹ {cls}"] = t["fares"][cls]["total"]
            else:
                base_row[f"₹ {cls}"] = None
        rows.append(base_row)
    
    if rows:
        df_fares = spark.createDataFrame(rows)
        display(df_fares)
    else:
        print("No direct trains found. Try major junction codes.")
else:
    print(f"Station not found. Check codes: FROM='{FROM_STATION}' TO='{TO_STATION}'")