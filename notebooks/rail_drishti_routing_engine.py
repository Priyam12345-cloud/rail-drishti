# Databricks notebook source
# DBTITLE 1,Rail-Drishti Architecture & Datasets
# MAGIC %md
# MAGIC ## Rail-Drishti: AI-Powered Indian Travel Intelligence
# MAGIC
# MAGIC ### AI-Centric Architecture
# MAGIC ```
# MAGIC                     "I missed my Rajdhani from Delhi to Mumbai"
# MAGIC                                       |
# MAGIC                           [AI LAYER 1: NLP Parser]
# MAGIC                           ai_query() extracts intent
# MAGIC                           origin=NDLS, dest=BCT, time=18:00
# MAGIC                                       |
# MAGIC                           [Train Data: Spark DataFrames]
# MAGIC                           Stations + Schedules + Fares
# MAGIC                           (loaded from CSV / Delta tables)
# MAGIC                                       |
# MAGIC                           [AI LAYER 2: Route Reasoner]
# MAGIC                           ai_query() ranks routes using:
# MAGIC                           - delay risk awareness
# MAGIC                           - waitlist confirmation probability
# MAGIC                           - cost vs time trade-offs
# MAGIC                           - connection safety margins
# MAGIC                                       |
# MAGIC                           [AI LAYER 3: Travel Advisor]
# MAGIC                           ai_query() generates natural language
# MAGIC                           recommendation with explanation
# MAGIC                                       |
# MAGIC                     "Take the 21:25 Golden Temple Mail (Rs.605).
# MAGIC                      The Duronto is faster but runs Wed/Sat only.
# MAGIC                      Via Kota adds 4hrs but costs Rs.400 less."
# MAGIC ```
# MAGIC
# MAGIC ### Where AI Runs (Databricks ai_query)
# MAGIC
# MAGIC | AI Layer | What It Does | Without AI | With AI |
# MAGIC |----------|-------------|------------|----------|
# MAGIC | **NLP Parser** | Understands "missed train from Delhi" | Hardcoded station codes | Natural language input |
# MAGIC | **Route Reasoner** | Picks best route from 20+ options | Sort by time only | Weighs delay risk, comfort, cost, connections |
# MAGIC | **Delay Predictor** | "This train is often 2hrs late" | No awareness | AI flags risky trains |
# MAGIC | **Waitlist Advisor** | "WL/45 = 23% chance" | Show raw WL number | AI predicts confirmation |
# MAGIC | **Travel Advisor** | Explains WHY this route | Raw data table | Conversational recommendation |

# COMMAND ----------

# DBTITLE 1,Dataset 1: Station Master Data
# MAGIC %md
# MAGIC ## Data: Datameet Indian Railways Dataset
# MAGIC
# MAGIC Loaded from `/Workspace/Users/lopamudra.wncc@gmail.com/Data/`
# MAGIC
# MAGIC | File | Records | Contents |
# MAGIC |------|---------|----------|
# MAGIC | `schedules.json` | 417,080 stops | Every stop of every train: arrival, departure, day, station |
# MAGIC | `stations.json` | 8,990 stations | Code, name, state, zone, lat/lng coordinates |
# MAGIC | `trains.json` | 5,208 trains | Train name, source/dest, classes (3AC, sleeper, etc.), route geometry |

# COMMAND ----------

# DBTITLE 1,Dataset 2: Train Schedule Data
import json
from datetime import datetime, timedelta
from collections import defaultdict
from pyspark.sql.types import *
from pyspark.sql import functions as F

DATA_PATH = "/Workspace/Users/lopamudra.wncc@gmail.com/Data"

# ============================================================
# 1. LOAD SCHEDULES (417K stops -- the main timetable)
# ============================================================
with open(f"{DATA_PATH}/schedules.json", "r") as f:
    schedules_raw = json.load(f)

df_schedule = spark.createDataFrame(schedules_raw)
df_schedule = (
    df_schedule
    .withColumn("train_number", F.col("train_number").cast("string"))
    .withColumn("day", F.col("day").cast("int"))
    .withColumn("arrival", F.regexp_replace(F.col("arrival"), "None", ""))
    .withColumn("departure", F.regexp_replace(F.col("departure"), "None", ""))
)

print(f"Schedules: {df_schedule.count()} stops")
print(f"Unique trains: {df_schedule.select('train_number').distinct().count()}")
display(df_schedule.limit(5))

# COMMAND ----------

# DBTITLE 1,Dataset 3-5: Fares, Buses, Flights
# ============================================================
# 2. LOAD STATIONS (8,990 stations with coordinates)
# ============================================================
with open(f"{DATA_PATH}/stations.json", "r") as f:
    stations_geo = json.load(f)

stations_list = []
for feat in stations_geo["features"]:
    props = feat["properties"]
    geom = feat.get("geometry")
    coords = geom["coordinates"] if geom and geom.get("coordinates") else [None, None]
    stations_list.append((
        props.get("code", ""),
        props.get("name", ""),
        props.get("state", ""),
        props.get("zone", ""),
        props.get("address", ""),
        coords[1] if coords[1] else None,
        coords[0] if coords[0] else None,
    ))

df_stations = spark.createDataFrame(
    stations_list,
    schema="station_code STRING, station_name STRING, state STRING, zone STRING, address STRING, latitude DOUBLE, longitude DOUBLE"
)

print(f"Stations: {df_stations.count()}")
display(df_stations.limit(5))

# ============================================================
# 3. LOAD TRAINS (5,208 trains with class availability)
# ============================================================
with open(f"{DATA_PATH}/trains.json", "r") as f:
    trains_geo = json.load(f)

trains_list = []
for feat in trains_geo["features"]:
    props = feat["properties"]
    trains_list.append((
        str(props.get("number", "")),
        props.get("name", ""),
        props.get("from_station_code", ""),
        props.get("from_station_name", ""),
        props.get("to_station_code", ""),
        props.get("to_station_name", ""),
        props.get("zone", ""),
        props.get("type", ""),
        int(props.get("third_ac", 0) or 0),
        int(props.get("sleeper", 0) or 0),
        int(props.get("second_ac", 0) or 0),
        int(props.get("first_ac", 0) or 0),
        int(props.get("first_class", 0) or 0),
        int(props.get("chair_car", 0) or 0),
        props.get("departure", ""),
        props.get("arrival", ""),
        props.get("duration_h", 0),
        props.get("duration_m", 0),
    ))

df_trains = spark.createDataFrame(
    trains_list,
    schema="""train_number STRING, train_name STRING, 
             from_station_code STRING, from_station_name STRING,
             to_station_code STRING, to_station_name STRING,
             zone STRING, train_type STRING,
             third_ac INT, sleeper INT, second_ac INT, first_ac INT, first_class INT, chair_car INT,
             departure STRING, arrival STRING, duration_h INT, duration_m INT"""
)

print(f"Trains: {df_trains.count()}")
display(df_trains.limit(5))

# COMMAND ----------

# DBTITLE 1,Core: Multi-Modal Routing Engine
# ============================================================
# Routing Engine -- graph + index-based with CITY CLUSTER support
# Finds both DIRECT trains and VIA-HUB connecting routes
# ============================================================

from datetime import datetime, timedelta

# --- City clusters: one code maps to ALL terminals in that city ---
CITY_CLUSTERS = {
    "BCT":["BCT","CSTM","LTT","BDTS","DR","DDR"],"CSTM":["BCT","CSTM","LTT","BDTS","DR","DDR"],
    "LTT":["BCT","CSTM","LTT","BDTS","DR","DDR"],"BDTS":["BCT","CSTM","LTT","BDTS","DR","DDR"],
    "DR":["BCT","CSTM","LTT","BDTS","DR","DDR"],"DDR":["BCT","CSTM","LTT","BDTS","DR","DDR"],
    "NDLS":["NDLS","DEE","NZM","ANVT","DLI"],"DEE":["NDLS","DEE","NZM","ANVT","DLI"],
    "NZM":["NDLS","DEE","NZM","ANVT","DLI"],"ANVT":["NDLS","DEE","NZM","ANVT","DLI"],
    "DLI":["NDLS","DEE","NZM","ANVT","DLI"],
    "HWH":["HWH","SDAH","KOAA"],"SDAH":["HWH","SDAH","KOAA"],"KOAA":["HWH","SDAH","KOAA"],
    "MAS":["MAS","MS","MSB","TBM"],"MS":["MAS","MS","MSB","TBM"],
    "SBC":["SBC","KSR","YPR","BNCE"],"KSR":["SBC","KSR","YPR","BNCE"],"YPR":["SBC","KSR","YPR","BNCE"],
    "SC":["SC","HYB","KCG"],"HYB":["SC","HYB","KCG"],"KCG":["SC","HYB","KCG"],
    "PNBE":["PNBE","PNC","RJPB","DNR"],"PNC":["PNBE","PNC","RJPB","DNR"],
    "RJPB":["PNBE","PNC","RJPB","DNR"],"DNR":["PNBE","PNC","RJPB","DNR"],
    "LKO":["LKO","LJN"],"LJN":["LKO","LJN"],
    "ADI":["ADI","SBIJ"],"JP":["JP","JPS"],
    "CNB":["CNB","CPNB"],"PUNE":["PUNE","PNVL"],
}

CITY_ALIASES = {
    "MUMBAI":"BCT","DELHI":"NDLS","KOLKATA":"HWH","CALCUTTA":"HWH",
    "CHENNAI":"MAS","MADRAS":"MAS","BANGALORE":"SBC","BENGALURU":"SBC",
    "HYDERABAD":"SC","PATNA":"PNBE","LUCKNOW":"LKO","AHMEDABAD":"ADI",
    "JAIPUR":"JP","KANPUR":"CNB","PUNE":"PUNE","VARANASI":"BSB","BHOPAL":"BPL",
}

def _expand_cluster(code):
    """Expand a station code to all equivalent terminals in that city."""
    return set(CITY_CLUSTERS.get(code, [code]))


class RailDrishtiRouter:
    """Graph + Index based routing engine with city cluster support."""

    def __init__(self):
        self.graph = defaultdict(list)
        self.hubs = set()
        self.station_names = {}
        self.train_stops = {}              # train_no -> [{code, name, dep, arr, day, train_name}, ...]
        self.station_to_trains = defaultdict(set)  # station_code -> set of train_nos

    def build_from_schedule(self, schedule_df):
        """Build graph + indexes from schedules DataFrame."""
        # Identify hubs: stations with 10+ trains
        station_counts = (
            schedule_df.groupBy("station_code")
            .agg(F.countDistinct("train_number").alias("train_count"))
            .filter("train_count >= 10")
            .collect()
        )
        for row in station_counts:
            self.hubs.add(row.station_code)

        # Station name lookup
        for row in schedule_df.select("station_code", "station_name").distinct().collect():
            if row.station_code and row.station_name:
                self.station_names[row.station_code] = row.station_name

        # Group schedule by train
        trains = defaultdict(list)
        for row in schedule_df.collect():
            trains[row.train_number].append(row)

        edge_count = 0
        for train_no, stops in trains.items():
            stops.sort(key=lambda x: (x.day or 0, x.id or 0))
            train_name = stops[0].train_name or train_no
            if len(stops) < 2:
                continue

            origin_stop = stops[0]
            dest_stop = stops[-1]

            # --- Build graph edges (for via-hub routing) ---
            for i, stop in enumerate(stops):
                if not stop.station_code:
                    continue
                if i > 0 and origin_stop.station_code and stop.station_code != origin_stop.station_code:
                    edge = {
                        "train_no": train_no, "train_name": train_name,
                        "from_stn": origin_stop.station_code, "to_stn": stop.station_code,
                        "departure": (origin_stop.departure or "").replace(":00", "", 1) if origin_stop.departure else "",
                        "arrival": (stop.arrival or "").replace(":00", "", 1) if stop.arrival else "",
                        "day_from": origin_stop.day or 1, "day_to": stop.day or 1,
                        "fare": 0, "fares": {},
                    }
                    self.graph[origin_stop.station_code].append((stop.station_code, edge))
                    edge_count += 1
                if i < len(stops) - 1 and dest_stop.station_code and stop.station_code != dest_stop.station_code:
                    edge = {
                        "train_no": train_no, "train_name": train_name,
                        "from_stn": stop.station_code, "to_stn": dest_stop.station_code,
                        "departure": (stop.departure or "").replace(":00", "", 1) if stop.departure else "",
                        "arrival": (dest_stop.arrival or "").replace(":00", "", 1) if dest_stop.arrival else "",
                        "day_from": stop.day or 1, "day_to": dest_stop.day or 1,
                        "fare": 0, "fares": {},
                    }
                    self.graph[stop.station_code].append((dest_stop.station_code, edge))
                    edge_count += 1
                if i < len(stops) - 1:
                    nxt = stops[i + 1]
                    if nxt.station_code and stop.station_code != nxt.station_code:
                        edge = {
                            "train_no": train_no, "train_name": train_name,
                            "from_stn": stop.station_code, "to_stn": nxt.station_code,
                            "departure": (stop.departure or "").replace(":00", "", 1) if stop.departure else "",
                            "arrival": (nxt.arrival or "").replace(":00", "", 1) if nxt.arrival else "",
                            "day_from": stop.day or 1, "day_to": nxt.day or 1,
                            "fare": 0, "fares": {},
                        }
                        self.graph[stop.station_code].append((nxt.station_code, edge))
                        edge_count += 1

            # --- Build indexes (for direct train search) ---
            ordered = []
            for s in stops:
                if s.station_code:
                    ordered.append({
                        "code": s.station_code,
                        "name": s.station_name or "",
                        "departure": (s.departure or "").replace(":00", "", 1) if s.departure else "",
                        "arrival": (s.arrival or "").replace(":00", "", 1) if s.arrival else "",
                        "day": s.day or 1,
                        "train_name": train_name,
                    })
                    self.station_to_trains[s.station_code].add(train_no)
            self.train_stops[train_no] = ordered

        print(f"Graph: {len(self.graph)} stations, {edge_count} edges, {len(self.hubs)} hubs, {len(trains)} trains")
        print(f"Indexes: {len(self.train_stops)} train stop-lists, {len(self.station_to_trains)} station->train mappings")

    # --- Time helpers (unchanged) ---
    def _time_to_mins(self, t, day=1):
        try:
            t = str(t).strip()
            parts = t.replace(".", ":").split(":")
            h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            return (day - 1) * 1440 + h * 60 + m
        except (ValueError, AttributeError, IndexError):
            return (day - 1) * 1440

    def _format_mins(self, mins):
        if mins <= 0: mins += 1440
        return f"{mins // 60}h {mins % 60}m"

    def _mins_to_datetime(self, mins, travel_date):
        if travel_date is None:
            return None
        return travel_date + timedelta(minutes=mins)

    def _fmt_dt(self, dt):
        if dt is None:
            return ""
        return dt.strftime("%b %d %H:%M")

    # --- DIRECT routes: index-based with city cluster expansion ---
    def find_direct_routes(self, origin, dest, depart_after="00:00", travel_date=None):
        """Find direct trains by intersecting trains at expanded origin & dest clusters.
        E.g., PNBE->BCT also finds trains going PNBE->LTT, PNBE->CSTM, PNBE->BDTS."""
        dep_min = self._time_to_mins(depart_after)
        ori_codes = _expand_cluster(origin)
        dst_codes = _expand_cluster(dest)

        # Get all trains passing through any origin cluster code
        ori_trains = set()
        for c in ori_codes:
            ori_trains |= self.station_to_trains.get(c, set())
        dst_trains = set()
        for c in dst_codes:
            dst_trains |= self.station_to_trains.get(c, set())
        common = ori_trains & dst_trains

        routes, seen = [], set()
        for tn in common:
            stops = self.train_stops.get(tn, [])
            codes = [s["code"] for s in stops]
            # Find first origin match
            fi, actual_from = None, None
            for i, c in enumerate(codes):
                if c in ori_codes:
                    fi, actual_from = i, c
                    break
            if fi is None:
                continue
            # Find first dest match AFTER origin
            ti, actual_to = None, None
            for j in range(fi + 1, len(codes)):
                if codes[j] in dst_codes:
                    ti, actual_to = j, codes[j]
                    break
            if ti is None:
                continue

            dep_stop = stops[fi]
            arr_stop = stops[ti]
            dep = self._time_to_mins(dep_stop["departure"], dep_stop["day"])
            arr = self._time_to_mins(arr_stop["arrival"], arr_stop["day"])
            if dep < dep_min:
                continue
            travel = arr - dep
            if travel <= 0:
                travel += 1440
            if tn in seen:
                continue
            seen.add(tn)
            dep_dt = self._mins_to_datetime(dep, travel_date)
            arr_dt = self._mins_to_datetime(dep + travel, travel_date)
            routes.append({
                "type": "DIRECT", "train_no": tn,
                "train_name": dep_stop.get("train_name", tn),
                "departure": dep_stop["departure"], "arrival": arr_stop["arrival"],
                "travel_time": self._format_mins(travel), "travel_mins": travel,
                "cheapest_fare": 0, "fares": {},
                "availability": "NOT_CHECKED", "legs": 1, "via": "--",
                "depart_date": self._fmt_dt(dep_dt),
                "arrive_date": self._fmt_dt(arr_dt),
                "actual_origin": actual_from,
                "actual_dest": actual_to,
            })
        routes.sort(key=lambda r: r["travel_mins"])
        return routes

    # --- VIA-HUB routes: cluster-aware connecting trains ---
    def find_via_hub_routes(self, origin, dest, depart_after="00:00",
                            max_hubs=10, min_conn=45, travel_date=None):
        dep_min = self._time_to_mins(depart_after)
        ori_codes = _expand_cluster(origin)
        dst_codes = _expand_cluster(dest)

        # Collect all edges FROM any origin-cluster code
        origin_edges = []
        for oc in ori_codes:
            origin_edges.extend(self.graph.get(oc, []))

        # Score hubs: must be reachable from origin AND have edges to dest cluster
        scored = []
        for h in self.hubs:
            if h in ori_codes or h in dst_codes:
                continue
            has_leg1 = any(d == h for d, _ in origin_edges)
            has_leg2 = any(d in dst_codes for d, _ in self.graph.get(h, []))
            if has_leg1 and has_leg2:
                scored.append(h)

        via_routes, seen = [], set()
        for hub in scored[:max_hubs]:
            for _, l1 in origin_edges:
                if l1["to_stn"] != hub:
                    continue
                l1d = self._time_to_mins(l1["departure"], l1.get("day_from", 1))
                if l1d < dep_min:
                    continue
                l1a = self._time_to_mins(l1["arrival"], l1.get("day_to", 1))
                if l1a <= l1d:
                    l1a += 1440

                for dest_stn, l2 in self.graph.get(hub, []):
                    if dest_stn not in dst_codes:
                        continue
                    l2_dep_own = self._time_to_mins(l2["departure"], l2.get("day_from", 1))
                    l2_arr_own = self._time_to_mins(l2["arrival"], l2.get("day_to", 1))
                    if l2_arr_own <= l2_dep_own:
                        l2_arr_own += 1440
                    leg2_duration = l2_arr_own - l2_dep_own

                    l2_dep_tod = self._time_to_mins(l2["departure"])
                    earliest_board = l1a + min_conn
                    l2d = l2_dep_tod
                    while l2d < earliest_board:
                        l2d += 1440
                    conn_wait = l2d - l1a
                    if conn_wait > 2880:
                        continue
                    l2a = l2d + leg2_duration
                    total = l2a - l1d
                    if total > 4320 or total < 0:
                        continue
                    key = f"{l1['train_no']}_{l2['train_no']}_{hub}"
                    if key in seen:
                        continue
                    seen.add(key)

                    dep_dt = self._mins_to_datetime(l1d, travel_date)
                    hub_arr_dt = self._mins_to_datetime(l1a, travel_date)
                    hub_dep_dt = self._mins_to_datetime(l2d, travel_date)
                    arr_dt = self._mins_to_datetime(l2a, travel_date)

                    via_routes.append({
                        "type": f"VIA {hub}",
                        "train_no": f"{l1['train_no']}+{l2['train_no']}",
                        "train_name": f"{l1['train_name']} -> {l2['train_name']}",
                        "departure": l1["departure"], "arrival": l2["arrival"],
                        "travel_time": self._format_mins(total), "travel_mins": total,
                        "cheapest_fare": 0, "fares": {},
                        "availability": "NOT_CHECKED", "legs": 2, "via": hub,
                        "connection_wait": self._format_mins(conn_wait),
                        "depart_date": self._fmt_dt(dep_dt),
                        "hub_arrive_date": self._fmt_dt(hub_arr_dt),
                        "hub_depart_date": self._fmt_dt(hub_dep_dt),
                        "arrive_date": self._fmt_dt(arr_dt),
                        "actual_origin": l1["from_stn"],
                        "actual_dest": dest_stn,
                    })

        via_routes.sort(key=lambda r: r["travel_mins"])
        return via_routes[:200]  # cap at 200

    def missed_train_protocol(self, origin, dest, current_time="00:00",
                               travel_date=None, budget_max=None, max_hub_search=10):
        # Resolve city aliases
        origin = CITY_ALIASES.get(origin.upper(), origin)
        dest = CITY_ALIASES.get(dest.upper(), dest)

        if travel_date is None:
            td = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif isinstance(travel_date, str):
            td = datetime.strptime(travel_date, "%Y-%m-%d")
        else:
            td = travel_date

        ori_name = self.station_names.get(origin, '')
        dst_name = self.station_names.get(dest, '')
        dst_cluster = _expand_cluster(dest)
        dst_label = f"{dest}" + (f" (+ {', '.join(c for c in dst_cluster if c != dest)})" if len(dst_cluster) > 1 else "")

        print(f"\n{'='*60}")
        print(f"MISSED TRAIN PROTOCOL")
        print(f"  {origin} ({ori_name}) -> {dst_label} ({dst_name})")
        print(f"  Travel Date: {td.strftime('%a %b %d, %Y')}  |  After: {current_time}")
        print(f"{'='*60}")

        opts = []
        direct = self.find_direct_routes(origin, dest, depart_after=current_time, travel_date=td)
        opts.extend(direct)
        print(f"  Direct: {len(direct)} trains")

        via = self.find_via_hub_routes(origin, dest, depart_after=current_time,
                                       max_hubs=max_hub_search, travel_date=td)
        opts.extend(via)
        print(f"  Via-hub: {len(via)} routes")

        opts.sort(key=lambda x: x["travel_mins"])
        print(f"  {'\u2500'*56}")
        for i, o in enumerate(opts[:10], 1):
            dep_dt = o.get('depart_date', '')
            arr_dt = o.get('arrive_date', '')
            actual_o = o.get('actual_origin', '')
            actual_d = o.get('actual_dest', '')
            stn_info = f" [{actual_o}\u2192{actual_d}]" if actual_o and actual_d else ""
            if o["legs"] == 2:
                hub_info = f" | Hub: arr {o.get('hub_arrive_date','')} dep {o.get('hub_depart_date','')} (wait {o.get('connection_wait','')})"
            else:
                hub_info = ""
            print(f"  [{i}] {o['type']}{stn_info} | {o['train_name']} ({o['train_no']})")
            print(f"      Dep: {dep_dt}  ->  Arr: {arr_dt}  |  {o['travel_time']}{hub_info}")
        return opts

print("RailDrishtiRouter ready (index-based + city clusters)")

# COMMAND ----------

# DBTITLE 1,Initialize Router with All Data
# ============================================================
# Build router from real 417K schedule entries
# ============================================================
router = RailDrishtiRouter()
router.build_from_schedule(df_schedule)

print(f"\nSample hubs: {list(router.hubs)[:15]}")
print(f"\nTry: router.missed_train_protocol('NDLS', 'BCT', current_time='18:00')")

# COMMAND ----------

# DBTITLE 1,Fare Calculation Engine (Indian Railways)
# ============================================================
# FARE CALCULATION ENGINE
# Distance-based Indian Railways pricing for ALL classes:
#   GEN | SL | CC | 3AC | 2AC | 1AC
# Uses haversine hop-by-hop distance along the train's route
# ============================================================
import math

# --- 1. Station coordinates (for distance calculation) ---
station_coords = {}  # code -> (lat, lng)
for feat in stations_geo["features"]:
    p = feat["properties"]
    g = feat.get("geometry")
    if g and g.get("coordinates"):
        c = g["coordinates"]
        if c[0] is not None and c[1] is not None:
            station_coords[p["code"]] = (c[1], c[0])  # lat, lng

# --- 2. Train type map (for premium train fare multipliers) ---
train_type_map = {}  # train_no -> type string
for feat in trains_geo["features"]:
    p = feat["properties"]
    train_type_map[str(p.get("number", ""))] = p.get("type", "Mail/Exp")

print(f"Station coords: {len(station_coords)} | Train types: {len(train_type_map)}")

# --- 3. Distance functions ---
def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance between two lat/lng points."""
    if any(v is None for v in (lat1, lng1, lat2, lng2)):
        return None
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def route_distance_km(train_no, from_code, to_code):
    """Compute route distance by summing hop-by-hop haversine along the train's actual stops."""
    stop_list = router.train_stops.get(train_no, [])
    if not stop_list:
        c1, c2 = station_coords.get(from_code), station_coords.get(to_code)
        if c1 and c2:
            d = haversine_km(c1[0], c1[1], c2[0], c2[1])
            return round(d * 1.3) if d else None
        return None
    # Find from/to indices
    from_idx, to_idx = None, None
    for i, s in enumerate(stop_list):
        if s["code"] == from_code and from_idx is None:
            from_idx = i
        if s["code"] == to_code and from_idx is not None:
            to_idx = i
            break
    if from_idx is None or to_idx is None or from_idx >= to_idx:
        c1, c2 = station_coords.get(from_code), station_coords.get(to_code)
        if c1 and c2:
            d = haversine_km(c1[0], c1[1], c2[0], c2[1])
            return round(d * 1.3) if d else None
        return None
    total = 0.0
    for i in range(from_idx, to_idx):
        c1 = station_coords.get(stop_list[i]["code"])
        c2 = station_coords.get(stop_list[i+1]["code"])
        if c1 and c2:
            seg = haversine_km(c1[0], c1[1], c2[0], c2[1])
            if seg:
                total += seg
    return round(total * 1.2) if total > 0 else None  # 1.2x for rail curvature

# --- 4. Indian Railways fare formula ---
PER_KM  = {"GEN": 0.22, "SL": 0.45, "CC": 1.10, "3AC": 1.20, "2AC": 1.85, "1AC": 3.10}
RSV_CHG = {"GEN": 0,    "SL": 40,   "CC": 40,   "3AC": 40,   "2AC": 50,   "1AC": 60}
SF_CHG  = {"GEN": 0,    "SL": 30,   "CC": 30,   "3AC": 45,   "2AC": 50,   "1AC": 75}
TYPE_MULT = {
    "Rajdhani":1.35, "Shatabdi":1.30, "Duronto":1.25, "Vande Bharat":1.40,
    "Garib Rath":0.75, "Tejas":1.45, "Humsafar":1.15, "Jan Shatabdi":1.10,
}
ALL_CLASSES = ["GEN", "SL", "CC", "3AC", "2AC", "1AC"]

def calc_fare(dist_km, cls, train_type="", train_no=""):
    """Calculate fare for one class, one leg."""
    if not dist_km or dist_km <= 0:
        return None
    base = round(dist_km * PER_KM.get(cls, 0.45))
    for kw, mult in TYPE_MULT.items():
        if kw.lower() in (train_type or "").lower():
            base = round(base * mult)
            break
    rsv = RSV_CHG.get(cls, 40)
    sf = SF_CHG.get(cls, 30) if train_no.startswith("12") or train_no.startswith("22") else 0
    return base + rsv + sf

# --- 5. Enrich route results with fares ---
def enrich_with_fares(results):
    """Add distance-based per-class fares to every route result."""
    for r in results:
        train_nos = r["train_no"].split("+")
        is_via = "VIA" in r.get("type", "")
        ori = r.get("actual_origin", "")
        dst = r.get("actual_dest", "")

        if is_via and len(train_nos) >= 2:
            hub = r.get("via", "")
            d1 = route_distance_km(train_nos[0], ori, hub) if ori and hub else None
            d2 = route_distance_km(train_nos[1], hub, dst) if hub and dst else None
            t1, t2 = train_type_map.get(train_nos[0], ""), train_type_map.get(train_nos[1], "")
            fares = {}
            for cls in ALL_CLASSES:
                f1 = calc_fare(d1, cls, t1, train_nos[0]) if d1 else 0
                f2 = calc_fare(d2, cls, t2, train_nos[1]) if d2 else 0
                if f1 or f2:
                    fares[cls] = (f1 or 0) + (f2 or 0)
                    fares[f"l1_{cls}"] = f1 or 0
                    fares[f"l2_{cls}"] = f2 or 0
            r["fares"] = fares
            r["distance_km"] = (d1 or 0) + (d2 or 0)
            r["leg1_dist"] = d1
            r["leg2_dist"] = d2
        else:
            tno = train_nos[0]
            dist = route_distance_km(tno, ori, dst) if ori and dst else None
            ttype = train_type_map.get(tno, "")
            fares = {}
            for cls in ALL_CLASSES:
                f = calc_fare(dist, cls, ttype, tno)
                if f:
                    fares[cls] = f
            r["fares"] = fares
            r["distance_km"] = dist
    return results

print("\n\u2705 Fare engine ready")
print(f"   Classes: {' | '.join(ALL_CLASSES)}")
print(f"   Sample: 1000km Sleeper = \u20b9{calc_fare(1000, 'SL', 'Mail/Exp', '12345')}")

# COMMAND ----------

# DBTITLE 1,Demo: Missed Train Delhi to Mumbai
# ============================================================
# DEMO: Patna -> Mumbai with FARES (like Amazon train booking)
# Shows direct + connecting trains with per-class pricing
# ============================================================

results = router.missed_train_protocol(
    origin="PNBE",
    dest="BCT",
    current_time="00:00",
    travel_date="2026-03-29",
    max_hub_search=12
)
results = enrich_with_fares(results)

direct = [r for r in results if "DIRECT" in r["type"]]
connecting = [r for r in results if "VIA" in r["type"]]

# Deduplicate connecting: best per (train1, hub)
seen_c = set()
best_conn = []
for r in connecting:
    k = (r["train_no"].split("+")[0], r.get("via",""))
    if k not in seen_c:
        seen_c.add(k)
        best_conn.append(r)
    if len(best_conn) >= 10: break

print(f"\n{'='*78}")
print(f"  PATNA JN (PNBE) \u2192 MUMBAI  |  {len(direct)} Direct  \u2022  {len(connecting)} Connecting")
print(f"{'='*78}")

# --- DIRECT TRAINS ---
print(f"\n  \u2708  DIRECT TRAINS ({len(direct)})")
print(f"  {'\u2500'*74}")
for r in direct[:12]:
    dst = r.get('actual_dest', '')
    dist = r.get('distance_km', 0) or 0
    fares = r.get('fares', {})
    ttype = train_type_map.get(r['train_no'], '')
    print(f"\n  \u250c{'\u2500'*74}\u2510")
    print(f"  \u2502  {r['train_no']} {r['train_name'][:38]:38s} {ttype[:12]:12s} {r['travel_time']:>8}  \u2502")
    print(f"  \u2502  Dep: {r.get('depart_date',''):>12}  \u2192  Arr: {r.get('arrive_date',''):>12}   {dist:>5} km      \u2502")
    if dst != 'BCT':
        dst_name = router.station_names.get(dst, dst)
        print(f"  \u2502  Terminal: {dst} ({dst_name[:30]}){' '*(37-len(dst_name[:30]))}  \u2502")
    if fares:
        print(f"  \u2502{'\u2500'*74}\u2502")
        parts = [f"{cls}:\u20b9{fares[cls]:,}" for cls in ALL_CLASSES if fares.get(cls, 0) > 0]
        line = ' \u2502 '.join(parts)
        print(f"  \u2502  {line:72s}  \u2502")
    print(f"  \u2514{'\u2500'*74}\u2518")

# --- CONNECTING TRAINS ---
print(f"\n  \U0001f504  CONNECTING TRAINS (top {len(best_conn)})")
print(f"  {'\u2500'*74}")
for r in best_conn[:6]:
    hub = r.get('via', '')
    hub_name = router.station_names.get(hub, hub)
    fares = r.get('fares', {})
    trains = r['train_no'].split('+')
    names = r['train_name'].split(' -> ') if ' -> ' in r['train_name'] else [r['train_name']]
    d1, d2 = r.get('leg1_dist', 0) or 0, r.get('leg2_dist', 0) or 0
    wait = r.get('connection_wait', '?')
    t1, t2 = trains[0] if trains else '?', trains[1] if len(trains)>1 else '?'
    n1 = names[0].strip() if names else '?'
    n2 = names[1].strip() if len(names)>1 else '?'
    print(f"\n  \u250c{'\u2500'*74}\u2510")
    print(f"  \u2502  VIA {hub_name[:18]} ({hub})  \u2022  {r['travel_time']:>8}  \u2022  {(d1+d2):,} km{' '*(25-len(str(d1+d2)))}  \u2502")
    print(f"  \u2502  Dep: {r.get('depart_date',''):>12}  \u2192  Arr: {r.get('arrive_date',''):>12}{' '*24}  \u2502")
    print(f"  \u2502{'\u2500'*74}\u2502")
    print(f"  \u2502  Leg 1: {t1} {n1[:33]:33s} {d1:>5} km          \u2502")
    print(f"  \u2502  \u21bb  Change at {hub} \u2022 Wait: {wait:22s}                 \u2502")
    print(f"  \u2502  Leg 2: {t2} {n2[:33]:33s} {d2:>5} km          \u2502")
    if fares:
        print(f"  \u2502{'\u2500'*74}\u2502")
        parts = [f"{cls}:\u20b9{fares[cls]:,}" for cls in ALL_CLASSES if fares.get(cls, 0) > 0]
        line = ' \u2502 '.join(parts)
        print(f"  \u2502  {line:72s}  \u2502")
    print(f"  \u2514{'\u2500'*74}\u2518")

print(f"\n  Total: {len(direct)} direct + {len(best_conn)} connecting routes found")

# COMMAND ----------

# DBTITLE 1,Display Results as Table
# ============================================================
# Summary DataFrame: all routes with fares
# ============================================================
rows = []
for r in direct[:12] + best_conn[:8]:
    f = r.get('fares', {})
    is_via = 'VIA' in r.get('type', '')
    rt = 'Non-Stop' if not is_via else f"Via {r.get('via','')}"
    dist = r.get('distance_km', 0) or 0
    rows.append((
        rt,
        r['train_name'][:45],
        r['train_no'],
        r.get('depart_date', ''),
        r.get('arrive_date', ''),
        r['travel_time'],
        dist,
        f.get('GEN', 0),
        f.get('SL', 0),
        f.get('3AC', 0),
        f.get('2AC', 0),
        f.get('1AC', 0),
    ))

schema = 'Route STRING, Train STRING, No STRING, Departs STRING, Arrives STRING, Duration STRING, Dist_km INT, GEN INT, SL INT, `3AC` INT, `2AC` INT, `1AC` INT'
df_display = spark.createDataFrame(rows, schema=schema)
print("Route Comparison with Fares:")
display(df_display)

# COMMAND ----------

# DBTITLE 1,AI Layer 1: Natural Language Query Parser
# ============================================================
# AI LAYER 1: Natural Language Travel Query Parser
# Uses Databricks ai_query() to understand user intent
# ============================================================

def ai_parse_travel_query(user_message: str) -> dict:
    """
    AI parses natural language into structured travel parameters.
    
    Input:  "I missed my Rajdhani from Delhi to Mumbai at 6pm"
    Output: {origin: "NDLS", destination: "BCT", time: "18:00", 
             scenario: "missed_train", train_mentioned: "12952"}
    """
    prompt = f"""You are an Indian Railways travel assistant. Parse the user's travel query into structured JSON.

IMPORTANT: You must respond with ONLY valid JSON, no other text.

Extract these fields:
- origin_station_code: The railway station code for the origin city (e.g., NDLS for New Delhi, BCT for Mumbai Central, HWH for Howrah). Use standard Indian railway codes.
- destination_station_code: Same for destination.
- origin_city: City name
- destination_city: City name  
- current_time: Time in HH:MM format (24hr). If "evening" use 18:00, "morning" 08:00, "night" 22:00.
- scenario: One of ["missed_train", "planning_trip", "looking_alternatives", "checking_status", "waitlist_query"]
- train_mentioned: Train number if mentioned (e.g., 12952 for Rajdhani), else null
- travel_class_preference: One of ["SL", "3AC", "2AC", "1AC", "any"] based on context
- budget_concern: true/false - is the user price sensitive?
- urgency: One of ["immediate", "today", "flexible"]

Common Indian station codes:
- Delhi: NDLS (New Delhi), NZM (Nizamuddin), DEE (Delhi Sarai Rohilla)
- Mumbai: BCT (Mumbai Central), CSTM (CSMT), LTT (Lokmanya Tilak)
- Kolkata: HWH (Howrah), SDAH (Sealdah)
- Chennai: MAS (Chennai Central)
- Bangalore: SBC (Bangalore City), KSR (KSR Bengaluru)
- Hyderabad: SC (Secunderabad)
- Ahmedabad: ADI
- Jaipur: JP
- Lucknow: LKO
- Varanasi: BSB
- Kota: KOTA
- Bhopal: BPL
- Pune: PUNE

User query: "{user_message}"

Respond with ONLY JSON:"""

    result = spark.sql(f"""
        SELECT ai_query(
            'databricks-meta-llama-3-3-70b-instruct',
            "{prompt.replace('"', "'")}",
            modelParameters => named_struct('temperature', 0.1, 'max_tokens', 500)
        ) AS parsed
    """).collect()[0]["parsed"]

    # Parse the AI response
    try:
        import re
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(result)
        print(f"  [AI PARSED] Successfully extracted travel intent")
        print(f"    Origin: {parsed.get('origin_station_code', '?')} ({parsed.get('origin_city', '?')})")
        print(f"    Destination: {parsed.get('destination_station_code', '?')} ({parsed.get('destination_city', '?')})")
        print(f"    Time: {parsed.get('current_time', '?')}")
        print(f"    Scenario: {parsed.get('scenario', '?')}")
        print(f"    Urgency: {parsed.get('urgency', '?')}")
        return parsed
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  [AI PARSE WARNING] Could not parse JSON, raw response: {result[:200]}")
        return {"raw_response": result, "error": str(e)}


print("AI Layer 1: NLP Travel Query Parser ready")
print("  ai_parse_travel_query('I missed my Rajdhani from Delhi to Mumbai')")

# COMMAND ----------

# DBTITLE 1,AI Layer 2: Route Recommender + Waitlist Advisor
# ============================================================
# AI LAYER 2: Intelligent Route Recommender
# Uses ai_query() to reason about which route is BEST, not just fastest
# ============================================================

def ai_recommend_route(route_options: list, user_context: dict) -> str:
    """
    AI analyzes route options and provides intelligent recommendation.
    Unlike simple sorting, AI considers:
    - Delay patterns ("Rajdhani is often late in monsoon")
    - Connection risk ("45 min layover at Kota is too tight")
    - Comfort trade-offs ("3AC Rajdhani vs SL on slower train")
    - Budget optimization ("Via Kota saves Rs.800 for 3 extra hours")
    - Calendar dates ("departs Mar 29, arrives Mar 30")
    """
    # Build route summary for AI -- now includes calendar dates
    routes_text = ""
    for i, r in enumerate(route_options[:10], 1):
        fare_str = f"Rs.{r['cheapest_fare']}" if r.get('cheapest_fare', 0) > 0 else "fare unknown"
        dep_date = r.get('depart_date', '')
        arr_date = r.get('arrive_date', '')
        date_str = f"| Departs: {dep_date} | Arrives: {arr_date}" if dep_date else f"| Departs: {r['departure']} | Arrives: {r['arrival']}"
        
        if r['legs'] == 2:
            hub_arr = r.get('hub_arrive_date', '')
            hub_dep = r.get('hub_depart_date', '')
            conn_wait = r.get('connection_wait', '')
            hub_str = f" | Hub {r.get('via','')}: arrive {hub_arr}, depart {hub_dep} (wait: {conn_wait})"
        else:
            hub_str = ""
        
        routes_text += (f"Route {i}: [{r['type']}] {r['train_name']} ({r['train_no']}) "
                       f"{date_str} "
                       f"| Total time: {r['travel_time']} | Fare: {fare_str} "
                       f"| Seats: {r.get('availability', 'unknown')} "
                       f"| Legs: {r['legs']}{hub_str}\n")

    scenario = user_context.get("scenario", "planning_trip")
    urgency = user_context.get("urgency", "today")
    budget_concern = user_context.get("budget_concern", False)
    origin = user_context.get("origin_city", "origin")
    dest = user_context.get("destination_city", "destination")
    travel_date = user_context.get("travel_date", "today")

    prompt = f"""You are Rail-Drishti, an expert Indian Railways travel advisor AI. 
A passenger needs help getting from {origin} to {dest}. 
Travel date: {travel_date}. Scenario: {scenario}. Urgency: {urgency}. Budget sensitive: {budget_concern}.

Here are the available route options with ACTUAL calendar dates:
{routes_text}

Provide a clear, helpful recommendation in 4-6 sentences. Be specific:
1. Which route you recommend FIRST and WHY (not just fastest - consider reliability, comfort, seat availability)
2. Mention the ACTUAL departure and arrival dates clearly (e.g., "departs Saturday Mar 29 at 17:35, arrives Sunday Mar 30")
3. For via-hub routes, mention the connection: which station, which date, how long the wait is
4. A backup option if the first choice fails
5. Any warnings (tight connections, known delay-prone trains, waitlist risks)

Speak like a knowledgeable Indian travel agent - practical, direct, helpful. Use train names, not just numbers.
Do NOT use bullet points. Write in flowing conversational paragraphs."""

    result = spark.sql(f"""
        SELECT ai_query(
            'databricks-meta-llama-3-3-70b-instruct',
            "{prompt.replace('"', "'")}",
            modelParameters => named_struct('temperature', 0.5, 'max_tokens', 600)
        ) AS recommendation
    """).collect()[0]["recommendation"]

    return result


def ai_assess_waitlist(waitlist_position: int, train_type: str, 
                       travel_class: str, days_to_travel: int) -> str:
    """
    AI predicts waitlist confirmation probability.
    """
    prompt = f"""You are an Indian Railways waitlist prediction expert.

A passenger has waitlist position WL/{waitlist_position} on a {train_type} train in {travel_class} class.
Travel is in {days_to_travel} days.

Based on typical Indian Railways patterns, estimate:
1. Confirmation probability (percentage)
2. Whether they should book this or look for alternatives
3. Key factors affecting confirmation (quota, charting time, cancellation patterns)

Be specific with the percentage and reasoning. Keep response to 3-4 sentences."""

    result = spark.sql(f"""
        SELECT ai_query(
            'databricks-meta-llama-3-3-70b-instruct',
            "{prompt.replace('"', "'")}",
            modelParameters => named_struct('temperature', 0.3, 'max_tokens', 300)
        ) AS assessment
    """).collect()[0]["assessment"]

    return result


print("AI Layer 2: Route Recommender + Waitlist Advisor ready")
print("  ai_recommend_route(routes, context)  -- smart route picker with dates")
print("  ai_assess_waitlist(45, 'Rajdhani', 'SL', 3) -- WL predictor")

# COMMAND ----------

# DBTITLE 1,AI Layer 3: Complete Rail-Drishti AI Advisor
# ============================================================
# AI LAYER 3: Complete AI-Powered Travel Advisor
# ============================================================

def lookup_station_code(city_or_name: str) -> str:
    """Look up station code from real data.
    Prioritizes schedule data (codes that match the routing graph)."""
    if not city_or_name:
        return None
    search = city_or_name.upper().strip()
    # 1. Try schedule data first (these codes match the graph)
    match = df_schedule.filter(
        F.upper(F.col("station_name")).contains(search)
    ).select("station_code").first()
    if match:
        return match.station_code
    # 2. Try schedule with partial match
    match = df_schedule.filter(
        F.upper(F.col("station_code")) == search
    ).select("station_code").first()
    if match:
        return match.station_code
    # 3. Try stations.json (may have different codes)
    match = df_stations.filter(
        F.upper(F.col("station_name")).contains(search) |
        F.upper(F.col("address")).contains(search)
    ).first()
    if match:
        # Cross-check against schedule
        sched_match = df_schedule.filter(
            F.col("station_code") == match.station_code
        ).first()
        if sched_match:
            return match.station_code
        # Try matching by name in schedule
        sched_match = df_schedule.filter(
            F.upper(F.col("station_name")).contains(match.station_name.upper())
        ).select("station_code").first()
        if sched_match:
            return sched_match.station_code
    return None


def rail_drishti_ai(user_message: str, travel_date: str = None) -> str:
    """
    THE MAIN AI FUNCTION - end-to-end intelligent travel advisor.
    User types natural language -> AI handles everything.
    
    Args:
        user_message: Natural language travel query
        travel_date: Travel date as 'YYYY-MM-DD'. Defaults to today.
    """
    # Resolve travel date
    if travel_date is None:
        travel_dt = datetime.now()
        travel_date_str = travel_dt.strftime("%Y-%m-%d")
    else:
        travel_dt = datetime.strptime(travel_date, "%Y-%m-%d")
        travel_date_str = travel_date
    travel_day_name = travel_dt.strftime("%A %b %d, %Y")

    print(f"{'=' * 60}")
    print(f"RAIL-DRISHTI AI ADVISOR")
    print(f"  User: {user_message}")
    print(f"  Travel Date: {travel_day_name}")
    print(f"{'=' * 60}")
    
    print(f"\n[STEP 1] AI parsing your query (ai_query)...")
    parsed = ai_parse_travel_query(user_message)
    
    if "error" in parsed:
        return "Sorry, I couldn't understand your query. Try: 'Chennai to Ujjain'"
    
    origin = parsed.get("origin_station_code")
    destination = parsed.get("destination_station_code")
    origin_city = parsed.get("origin_city", "")
    dest_city = parsed.get("destination_city", "")
    current_time = parsed.get("current_time", "08:00")
    scenario = parsed.get("scenario", "planning_trip")
    
    # Lookup missing codes from real schedule data
    if not origin or origin == "None":
        origin = lookup_station_code(origin_city)
        if origin:
            print(f"    Resolved '{origin_city}' -> {origin}")
        else:
            return f"Could not find station for '{origin_city}'."
    else:
        # Verify AI's code exists in schedule data
        verify = df_schedule.filter(F.col("station_code") == origin).first()
        if not verify:
            resolved = lookup_station_code(origin_city)
            if resolved:
                print(f"    Corrected {origin} -> {resolved}")
                origin = resolved
    
    if not destination or destination == "None":
        destination = lookup_station_code(dest_city)
        if destination:
            print(f"    Resolved '{dest_city}' -> {destination}")
        else:
            return f"Could not find station for '{dest_city}'."
    else:
        verify = df_schedule.filter(F.col("station_code") == destination).first()
        if not verify:
            resolved = lookup_station_code(dest_city)
            if resolved:
                print(f"    Corrected {destination} -> {resolved}")
                destination = resolved
    
    print(f"\n[STEP 2] Searching routes: {origin} -> {destination} on {travel_day_name}...")
    
    if scenario == "waitlist_query":
        wl_assessment = ai_assess_waitlist(
            waitlist_position=45,
            train_type=parsed.get("train_type", "Express"),
            travel_class=parsed.get("travel_class_preference", "SL"),
            days_to_travel=3
        )
        print(f"\n[AI ASSESSMENT]\n{wl_assessment}")
        return wl_assessment
    
    route_options = router.missed_train_protocol(
        origin=origin,
        dest=destination,
        current_time=current_time,
        travel_date=travel_date_str,
        budget_max=None,
        max_hub_search=10
    )
    
    if not route_options:
        return f"No routes found from {origin} to {destination} after {current_time}."
    
    # Add travel_date to context for AI recommender
    parsed["travel_date"] = travel_day_name
    
    print(f"\n[STEP 3] AI analyzing {len(route_options)} options (ai_query)...")
    recommendation = ai_recommend_route(route_options, parsed)
    
    print(f"\n{'=' * 60}")
    print(f"RAIL-DRISHTI AI RECOMMENDATION")
    print(f"{'=' * 60}")
    print(f"\n{recommendation}")
    print(f"\n  Routes analyzed: {len(route_options)}")
    
    return recommendation


print("Rail-Drishti AI Advisor ready!")
print("  rail_drishti_ai('Chennai to Ujjain', travel_date='2026-03-29')")

# COMMAND ----------

# DBTITLE 1,Demo: AI-Powered Missed Train Advisor
# ============================================================
# DEMO: Chennai to Ujjain -- full AI-powered flow with DATE tracking
# User just types natural language, AI handles everything:
#   1. ai_query() parses "Chennai to Ujjain" -> MAS, UJN
#   2. Router searches 417K real schedule entries with DATE alignment
#   3. ai_query() recommends the best route with actual calendar dates
# ============================================================

recommendation = rail_drishti_ai(
    "I want to go from Patna to Mumbai, what are my options?",
    travel_date="2026-03-29"  # Today's date
)

# COMMAND ----------

# DBTITLE 1,Demo: AI Waitlist Prediction + More Queries
# ============================================================
# DEMO 2: More AI queries you can try
# ============================================================

# --- Waitlist prediction ---
print("\n" + "=" * 60)
print("DEMO 2: Waitlist Prediction")
print("=" * 60)
wl_result = ai_assess_waitlist(
    waitlist_position=45,
    train_type="Rajdhani",
    travel_class="3AC",
    days_to_travel=5
)
print(f"\nAI Assessment:\n{wl_result}")

# --- Try more queries: ---
# rail_drishti_ai("cheapest way to reach Mumbai from Jaipur")
# rail_drishti_ai("I need to reach Kolkata from Delhi by tomorrow morning")
# rail_drishti_ai("any train from Bhopal to Pune tonight?")

# COMMAND ----------

# DBTITLE 1,Load & Clean Bus Data + City-Station Mapping
# ============================================================
# 4. LOAD & CLEAN PAN-INDIA BUS ROUTES
# ============================================================
import pandas as pd
import re
from datetime import datetime, timedelta

BUS_CSV = "/Workspace/Users/lopamudra.wncc@gmail.com/Pan-India_Bus_Routes.csv"
df_bus_pd = pd.read_csv(BUS_CSV)
print(f"Raw bus routes: {len(df_bus_pd):,}")

# --- Add cost column: distance * 1.7 ---
df_bus_pd["Cost"] = (df_bus_pd["Distance"] * 1.7).round(0).astype(int)

# --- Parse Duration "D:H:M" -> total minutes ---
def parse_duration_to_mins(dur_str):
    try:
        parts = str(dur_str).split(":")
        if len(parts) == 3:
            days, hours, mins = int(parts[0]), int(parts[1]), int(float(parts[2]))
            return days * 1440 + hours * 60 + mins
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(float(parts[1]))
    except:
        return None
    return None

df_bus_pd["duration_mins"] = df_bus_pd["Duration"].apply(parse_duration_to_mins)

# --- Parse Departure/Arrival "HH:MM:SS AM/PM" -> "HH:MM" (24h) ---
def parse_time_to_24h(t_str):
    try:
        t_str = str(t_str).strip()
        for fmt in ["%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"]:
            try:
                dt = datetime.strptime(t_str, fmt)
                return dt.strftime("%H:%M")
            except ValueError:
                continue
    except:
        pass
    return None

df_bus_pd["dep_24h"] = df_bus_pd["Departure"].apply(parse_time_to_24h)
df_bus_pd["arr_24h"] = df_bus_pd["Arrival"].apply(parse_time_to_24h)

# --- Clean: drop rows with bad parses or zero/negative duration ---
before = len(df_bus_pd)
df_bus_pd = df_bus_pd.dropna(subset=["dep_24h", "arr_24h", "duration_mins"])
df_bus_pd = df_bus_pd[df_bus_pd["duration_mins"] > 0]
df_bus_pd = df_bus_pd[df_bus_pd["Distance"] > 0]
df_bus_pd["duration_mins"] = df_bus_pd["duration_mins"].astype(int)
print(f"After cleaning: {len(df_bus_pd):,} routes ({before - len(df_bus_pd)} dropped)")

# --- Normalize city names for matching ---
df_bus_pd["from_clean"] = df_bus_pd["From"].str.strip().str.upper()
df_bus_pd["to_clean"] = df_bus_pd["To"].str.strip().str.upper()

# ============================================================
# 5. BUILD CITY <-> STATION CODE MAPPING
#    Match bus city names to train station names/cities
# ============================================================
station_rows = df_stations.select("station_code", "station_name", "address").distinct().collect()

city_to_station = {}  # "CHENNAI" -> "MAS"
for row in station_rows:
    code = row.station_code
    name = (row.station_name or "").strip().upper()
    addr = (row.address or "").strip().upper()
    # Map full station name and common city tokens
    for label in [name, addr]:
        if label:
            city_to_station[label] = code
            # Also map first word (city core): "NEW DELHI" -> "NDLS" already,
            # but "CHENNAI CENTRAL" -> map "CHENNAI" too
            for token in label.split():
                if len(token) >= 4 and token not in city_to_station:
                    city_to_station[token] = code

# Also add schedule station_name -> code for comprehensive coverage
for row in df_schedule.select("station_code", "station_name").distinct().collect():
    name = (row.station_name or "").strip().upper()
    if name and name not in city_to_station:
        city_to_station[name] = row.station_code
    for token in name.split():
        if len(token) >= 4 and token not in city_to_station:
            city_to_station[token] = row.station_code

bus_cities = set(df_bus_pd["from_clean"].unique()) | set(df_bus_pd["to_clean"].unique())
matched = sum(1 for c in bus_cities if c in city_to_station)
print(f"\nCity-to-Station mapping: {len(city_to_station):,} entries")
print(f"Bus cities matched to train stations: {matched}/{len(bus_cities)} ({100*matched/len(bus_cities):.1f}%)")

print(f"\n--- Sample cleaned bus data ---")
display(spark.createDataFrame(df_bus_pd.head(10)))

# COMMAND ----------

# DBTITLE 1,Multi-Modal Routing Engine (Train + Bus)
# ============================================================
# MULTI-MODAL ROUTING ENGINE: Train + Bus + Mixed
# ============================================================
# Constraints:
#   - Max 3 transfers (4 legs)
#   - >=45 min buffer for mode switches (train<->bus)
#   - Night-transfer penalty (22:00-06:00)
#   - Returns fastest valid route across all mode combos
# ============================================================

from collections import defaultdict
from datetime import datetime, timedelta

class MultiModalRouter:
    NIGHT_START = 22 * 60
    NIGHT_END   = 6  * 60
    NIGHT_PENALTY = 120
    MODE_SWITCH_BUF = 45
    SAME_MODE_BUF   = 30
    MAX_WAIT = 2 * 1440

    def __init__(self):
        self.graph = defaultdict(list)
        self.train_hubs = set()
        self.bus_hubs = set()
        self.all_hubs = set()
        self.station_names = {}

    # ---- BUILD ----
    def build(self, train_router, bus_df, city_to_stn):
        train_ct = 0
        for stn, edges in train_router.graph.items():
            for dest, edge in edges:
                e = dict(edge); e["mode"] = "train"
                self.graph[stn].append((dest, e))
                train_ct += 1
        self.train_hubs = set(train_router.hubs)
        self.station_names = dict(train_router.station_names)

        bus_ct = 0
        bus_stn_cnt = defaultdict(int)
        for _, row in bus_df.iterrows():
            fs = city_to_stn.get(row["from_clean"])
            ts = city_to_stn.get(row["to_clean"])
            if not fs or not ts or fs == ts: continue
            dep, arr, dur = row.get("dep_24h",""), row.get("arr_24h",""), row.get("duration_mins",0)
            if not dep or not arr or dur <= 0: continue
            dep_m = self._t2m(dep)
            day_to = 1 + (dep_m + dur) // 1440
            op = str(row.get("Operator","Bus"))[:30]
            edge = {
                "train_no": f"BUS-{op.replace(' ','_')[:12]}",
                "train_name": f"BUS {op}",
                "mode": "bus", "from_stn": fs, "to_stn": ts,
                "departure": dep, "arrival": arr,
                "day_from": 1, "day_to": day_to,
                "fare": int(row.get("Cost", 0)),
                "fares": {"bus": int(row.get("Cost", 0))},
                "duration_mins": dur,
            }
            self.graph[fs].append((ts, edge))
            bus_ct += 1
            bus_stn_cnt[fs] += 1
            bus_stn_cnt[ts] += 1

        self.bus_hubs = {s for s, c in bus_stn_cnt.items() if c >= 5}
        self.all_hubs = self.train_hubs | self.bus_hubs
        print(f"Multi-Modal Graph: {train_ct:,} train + {bus_ct:,} bus edges")
        print(f"  {len(self.graph):,} stations | {len(self.all_hubs)} hubs ({len(self.train_hubs)} train + {len(self.bus_hubs)} bus)")

    # ---- HELPERS ----
    def _t2m(self, t, day=1):
        try:
            parts = str(t).strip().replace(".",":").split(":")
            return (day-1)*1440 + int(parts[0])*60 + (int(parts[1]) if len(parts)>1 else 0)
        except: return (day-1)*1440

    def _fmt(self, mins):
        if mins <= 0: mins += 1440
        d, h, m = mins//1440, (mins%1440)//60, mins%60
        return f"{d}d {h}h {m}m" if d else f"{h}h {m}m"

    def _dt(self, mins, td): return (td + timedelta(minutes=mins)) if td else None
    def _fdt(self, dt): return dt.strftime("%b %d %H:%M") if dt else ""
    def _is_night(self, mins):
        tod = mins % 1440
        return tod >= self.NIGHT_START or tod < self.NIGHT_END
    def _buf(self, m1, m2): return self.MODE_SWITCH_BUF if m1 != m2 else self.SAME_MODE_BUF

    def _align_leg(self, prev_arr, leg, prev_mode):
        """Align leg departure after prev_arr+buffer. Returns (dep, arr, wait) or None."""
        d_own = self._t2m(leg["departure"], leg.get("day_from",1))
        a_own = self._t2m(leg["arrival"], leg.get("day_to",1))
        if a_own <= d_own: a_own += 1440
        dur = a_own - d_own
        buf = self._buf(prev_mode, leg["mode"])
        earliest = prev_arr + buf
        d_tod = self._t2m(leg["departure"])
        d = d_tod
        while d < earliest: d += 1440
        wait = d - prev_arr
        if wait > self.MAX_WAIT: return None
        return (d, d + dur, wait)

    def _mk(self, rtype, modes, legs_info, total, score, fare, changes, via, td, nxf, waits=None):
        mode_str = " > ".join("BUS" if m=="bus" else "TRAIN" for m in modes)
        r = {
            "type": f"{rtype} [{mode_str}]",
            "mode_sequence": modes,
            "train_no": "+".join(li[0]["train_no"] for li in legs_info),
            "train_name": " -> ".join(li[0]["train_name"] for li in legs_info),
            "departure": legs_info[0][0]["departure"],
            "arrival": legs_info[-1][0].get("arrival",""),
            "travel_time": self._fmt(total), "travel_mins": total, "score": score,
            "cheapest_fare": fare, "fares": {}, "changes": changes, "via": via,
            "depart_date": self._fdt(self._dt(legs_info[0][1], td)),
            "arrive_date": self._fdt(self._dt(legs_info[-1][2], td)),
            "night_transfers": nxf,
            "legs": [li[0] for li in legs_info],
        }
        if waits: r["connection_waits"] = [self._fmt(w) for w in waits]
        return r

    # ---- DIRECT (0 changes) ----
    def _find_direct(self, origin, dest, dep_min, td):
        routes, seen = [], set()
        for d, e in self.graph.get(origin, []):
            if d != dest: continue
            dep = self._t2m(e["departure"], e.get("day_from",1))
            arr = self._t2m(e["arrival"], e.get("day_to",1))
            if dep < dep_min: continue
            travel = arr - dep
            if travel <= 0: travel += 1440
            if travel > 4320: continue
            key = f"{e['mode']}_{e['train_no']}_{dep}"
            if key in seen: continue
            seen.add(key)
            icon = "DIRECT BUS" if e["mode"]=="bus" else "DIRECT TRAIN"
            routes.append(self._mk(
                icon, [e["mode"]], [(e, dep, dep+travel)],
                travel, travel, e.get("fare",0), 0, "--", td, 0
            ))
        return sorted(routes, key=lambda r: r["score"])

    # ---- 1-CHANGE ----
    def _find_1change(self, origin, dest, dep_min, td, max_hubs=15):
        routes, seen = [], set()
        reachable = {d for d,_ in self.graph.get(origin,[]) if d != dest}
        cands = [h for h in reachable if any(d==dest for d,_ in self.graph.get(h,[]))]
        cands.sort(key=lambda h: (0 if h in self.all_hubs else 1))
        for hub in cands[:max_hubs]:
            for _, l1 in self.graph.get(origin, []):
                if l1["to_stn"] != hub: continue
                l1d = self._t2m(l1["departure"], l1.get("day_from",1))
                if l1d < dep_min: continue
                l1a = self._t2m(l1["arrival"], l1.get("day_to",1))
                if l1a <= l1d: l1a += 1440
                for _, l2 in self.graph.get(hub, []):
                    if l2["to_stn"] != dest: continue
                    aligned = self._align_leg(l1a, l2, l1["mode"])
                    if not aligned: continue
                    l2d, l2a, wait = aligned
                    total = l2a - l1d
                    if total > 4320 or total < 0: continue
                    key = f"{l1['train_no']}_{l2['train_no']}_{hub}"
                    if key in seen: continue
                    seen.add(key)
                    np1 = self.NIGHT_PENALTY if self._is_night(l1a) else 0
                    routes.append(self._mk(
                        f"VIA {hub}", [l1["mode"], l2["mode"]],
                        [(l1,l1d,l1a),(l2,l2d,l2a)],
                        total, total+np1, l1.get("fare",0)+l2.get("fare",0),
                        1, hub, td, 1 if np1 else 0, [wait]
                    ))
        return sorted(routes, key=lambda r: r["score"])

    # ---- 2-CHANGE ----
    def _find_2change(self, origin, dest, dep_min, td, max_h=8):
        routes, seen = [], set()
        h1s = [d for d,_ in self.graph.get(origin,[]) if d in self.all_hubs and d!=dest][:max_h]
        for h1 in h1s:
            h2s = set()
            for d,_ in self.graph.get(h1,[]):
                if d!=origin and d!=dest and d!=h1 and any(dd==dest for dd,_ in self.graph.get(d,[])):
                    h2s.add(d)
            for h2 in list(h2s)[:max_h]:
                best = self._best_3leg(origin, h1, h2, dest, dep_min, td)
                if best:
                    key = f"{best['train_no']}_{h1}_{h2}"
                    if key not in seen: seen.add(key); routes.append(best)
        return sorted(routes, key=lambda r: r["score"])[:20]

    def _best_3leg(self, o, h1, h2, d, dep_min, td):
        best = None
        for _, l1 in self.graph.get(o, []):
            if l1["to_stn"]!=h1: continue
            l1d = self._t2m(l1["departure"], l1.get("day_from",1))
            if l1d < dep_min: continue
            l1a = self._t2m(l1["arrival"], l1.get("day_to",1))
            if l1a <= l1d: l1a += 1440
            for _, l2 in self.graph.get(h1, []):
                if l2["to_stn"]!=h2: continue
                a2 = self._align_leg(l1a, l2, l1["mode"])
                if not a2: continue
                l2d, l2a, w1 = a2
                for _, l3 in self.graph.get(h2, []):
                    if l3["to_stn"]!=d: continue
                    a3 = self._align_leg(l2a, l3, l2["mode"])
                    if not a3: continue
                    l3d, l3a, w2 = a3
                    total = l3a - l1d
                    if total > 5760 or total < 0: continue
                    np1 = self.NIGHT_PENALTY if self._is_night(l1a) else 0
                    np2 = self.NIGHT_PENALTY if self._is_night(l2a) else 0
                    score = total + np1 + np2
                    if best is None or score < best["score"]:
                        best = self._mk(
                            f"VIA {h1} > {h2}", [l1["mode"],l2["mode"],l3["mode"]],
                            [(l1,l1d,l1a),(l2,l2d,l2a),(l3,l3d,l3a)],
                            total, score, sum(lg.get("fare",0) for lg in [l1,l2,l3]),
                            2, f"{h1} > {h2}", td, sum(1 for p in [np1,np2] if p), [w1,w2]
                        )
        return best

    # ---- 3-CHANGE (aggressive pruning) ----
    def _find_3change(self, origin, dest, dep_min, td, max_h=5):
        routes, seen = [], set()
        h1s = [d for d,_ in self.graph.get(origin,[]) if d in self.all_hubs and d!=dest][:max_h]
        for h1 in h1s:
            h2s = [d for d,_ in self.graph.get(h1,[]) if d in self.all_hubs and d!=origin and d!=dest and d!=h1][:max_h]
            for h2 in h2s:
                h3s = [d for d,_ in self.graph.get(h2,[]) if d!=origin and d!=h1 and d!=h2 and any(dd==dest for dd,_ in self.graph.get(d,[]))][:max_h]
                for h3 in h3s:
                    best = self._best_4leg(origin, h1, h2, h3, dest, dep_min, td)
                    if best:
                        key = best["train_no"]
                        if key not in seen: seen.add(key); routes.append(best)
        return sorted(routes, key=lambda r: r["score"])[:10]

    def _best_4leg(self, o, h1, h2, h3, d, dep_min, td):
        best = None
        for _, l1 in self.graph.get(o, []):
            if l1["to_stn"]!=h1: continue
            l1d = self._t2m(l1["departure"], l1.get("day_from",1))
            if l1d < dep_min: continue
            l1a = self._t2m(l1["arrival"], l1.get("day_to",1))
            if l1a <= l1d: l1a += 1440
            for _, l2 in self.graph.get(h1, []):
                if l2["to_stn"]!=h2: continue
                a2 = self._align_leg(l1a, l2, l1["mode"])
                if not a2: continue
                l2d, l2a, w1 = a2
                for _, l3 in self.graph.get(h2, []):
                    if l3["to_stn"]!=h3: continue
                    a3 = self._align_leg(l2a, l3, l2["mode"])
                    if not a3: continue
                    l3d, l3a, w2 = a3
                    for _, l4 in self.graph.get(h3, []):
                        if l4["to_stn"]!=d: continue
                        a4 = self._align_leg(l3a, l4, l3["mode"])
                        if not a4: continue
                        l4d, l4a, w3 = a4
                        total = l4a - l1d
                        if total > 7200 or total < 0: continue
                        np1 = self.NIGHT_PENALTY if self._is_night(l1a) else 0
                        np2 = self.NIGHT_PENALTY if self._is_night(l2a) else 0
                        np3 = self.NIGHT_PENALTY if self._is_night(l3a) else 0
                        score = total + np1 + np2 + np3
                        if best is None or score < best["score"]:
                            best = self._mk(
                                f"VIA {h1}>{h2}>{h3}",
                                [l1["mode"],l2["mode"],l3["mode"],l4["mode"]],
                                [(l1,l1d,l1a),(l2,l2d,l2a),(l3,l3d,l3a),(l4,l4d,l4a)],
                                total, score, sum(lg.get("fare",0) for lg in [l1,l2,l3,l4]),
                                3, f"{h1}>{h2}>{h3}", td, sum(1 for p in [np1,np2,np3] if p), [w1,w2,w3]
                            )
        return best

    # ---- MAIN ENTRY POINT ----
    def find_routes(self, origin, dest, depart_after="00:00", travel_date=None,
                    max_changes=3, max_results=15):
        if travel_date is None:
            td = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
        elif isinstance(travel_date, str):
            td = datetime.strptime(travel_date, "%Y-%m-%d")
        else: td = travel_date
        dep_min = self._t2m(depart_after)

        print(f"\n{'='*65}")
        print(f"  MULTI-MODAL ROUTE SEARCH (Train + Bus)")
        print(f"  {origin} ({self.station_names.get(origin,'')}) -> {dest} ({self.station_names.get(dest,'')})")
        print(f"  Date: {td.strftime('%a %b %d, %Y')} | After: {depart_after} | Max changes: {max_changes}")
        print(f"{'='*65}")

        all_r = []
        direct = self._find_direct(origin, dest, dep_min, td)
        all_r.extend(direct)
        t_d = sum(1 for r in direct if r["mode_sequence"]==["train"])
        b_d = sum(1 for r in direct if r["mode_sequence"]==["bus"])
        print(f"  Direct: {len(direct)} (TRAIN {t_d}, BUS {b_d})")

        if max_changes >= 1:
            one = self._find_1change(origin, dest, dep_min, td)
            all_r.extend(one); print(f"  1-change: {len(one)} routes")
        if max_changes >= 2:
            two = self._find_2change(origin, dest, dep_min, td)
            all_r.extend(two); print(f"  2-change: {len(two)} routes")
        if max_changes >= 3 and len(all_r) < 5:
            three = self._find_3change(origin, dest, dep_min, td)
            all_r.extend(three); print(f"  3-change: {len(three)} routes")
        elif max_changes >= 3:
            print(f"  3-change: skipped (enough routes)")

        all_r.sort(key=lambda r: r["score"])
        top = all_r[:max_results]

        print(f"\n  {'_'*61}")
        print(f"  TOP {len(top)} ROUTES (ranked: travel time + night penalty)")
        print(f"  {'_'*61}")
        for i, r in enumerate(top, 1):
            modes = " > ".join("BUS" if m=="bus" else "TRAIN" for m in r["mode_sequence"])
            nw = f" [!{r['night_transfers']} night xfer]" if r.get("night_transfers",0) else ""
            fare_s = f"Rs.{r['cheapest_fare']}" if r.get("cheapest_fare",0) > 0 else ""
            print(f"\n  [{i}] {r['type']}")
            print(f"      {r['train_name']}")
            print(f"      Dep: {r['depart_date']}  ->  Arr: {r['arrive_date']}  |  {r['travel_time']}  {fare_s}")
            if r["changes"] > 0:
                waits = ", ".join(r.get("connection_waits",[])) if r.get("connection_waits") else ""
                print(f"      Changes: {r['changes']} | Via: {r['via']} | Waits: {waits}{nw}")
        return top

print("MultiModalRouter class ready")

# COMMAND ----------

# DBTITLE 1,Build Multi-Modal Router (Train + Bus Graph)
# ============================================================
# Build combined Train + Bus graph
# Uses: router (from Cell 6), df_bus_pd & city_to_station (from Cell 14)
# ============================================================

mm_router = MultiModalRouter()
mm_router.build(router, df_bus_pd, city_to_station)

print(f"\nSample bus hubs: {list(mm_router.bus_hubs)[:15]}")
print(f"Sample train hubs: {list(mm_router.train_hubs)[:15]}")

# COMMAND ----------

# DBTITLE 1,Demo: Multi-Modal Delhi to Mumbai
# ============================================================
# DEMO: Multi-modal Delhi -> Mumbai
# Finds fastest route: train-only, bus-only, or mixed
# ============================================================

results = mm_router.find_routes(
    origin="BOM",
    dest="NGP",
    depart_after="18:00",
    travel_date="2026-03-29",
    max_changes=3,
    max_results=10
)

print(f"\n{'='*65}")
print(f"FASTEST OVERALL: {results[0]['type']}")
print(f"  {results[0]['train_name']}")
print(f"  {results[0]['depart_date']} -> {results[0]['arrive_date']}  |  {results[0]['travel_time']}")
if results[0]['cheapest_fare'] > 0:
    print(f"  Fare: Rs.{results[0]['cheapest_fare']}")

# Show mode breakdown
for mode_type in ['train', 'bus']:
    mode_routes = [r for r in results if all(m==mode_type for m in r['mode_sequence'])]
    if mode_routes:
        best = mode_routes[0]
        label = 'TRAIN' if mode_type=='train' else 'BUS'
        print(f"\nFASTEST {label}-ONLY: {best['train_name']} | {best['travel_time']}")

mixed = [r for r in results if len(set(r['mode_sequence'])) > 1]
if mixed:
    best = mixed[0]
    print(f"\nFASTEST MIXED: {best['train_name']} | {best['travel_time']} | Changes: {best['changes']}")
else:
    print(f"\nNo mixed (train+bus) routes found for this pair")