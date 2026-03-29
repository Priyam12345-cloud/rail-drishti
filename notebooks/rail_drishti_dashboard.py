# Databricks notebook source
# DBTITLE 1,Rail Drishti Dashboard
# MAGIC %md
# MAGIC # 🚂 Rail-Drishti: AI-Powered Indian Railways Dashboard
# MAGIC
# MAGIC > **India's Smartest Train Travel Intelligence Platform**
# MAGIC
# MAGIC | Feature | Description |
# MAGIC |---------|-------------|
# MAGIC | 🤖 **AI Travel Assistant** | Natural language chatbot for route planning, missed trains & passenger info |
# MAGIC | 🗺️ **Route Planner** | Graph-based routing across 5,208 trains & 8,533 stations |
# MAGIC | ⭐ **Train Ratings** | Community-driven train rating system with persistent storage |
# MAGIC | 📊 **Analytics** | Zone distribution, class availability & network insights |
# MAGIC | 📚 **Passenger Guide** | PDF-powered knowledge base (rules, tatkal FAQ, citizen charter) |

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install gradio PyPDF2 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Data and Build Routing Engine
import gradio as gr
import json
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
from pyspark.sql.types import *
from pyspark.sql import functions as F
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import PyPDF2

w = WorkspaceClient()

DATA_PATH = "/Workspace/Users/lopamudra.wncc@gmail.com/Data"
RATINGS_FILE = "/Workspace/Users/lopamudra.wncc@gmail.com/UI/train_ratings.json"

# ── Load Schedules (417K stops) ──
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
print(f"Schedules: {df_schedule.count()} stops | Trains: {df_schedule.select('train_number').distinct().count()}")

# ── Load Stations (8,990 with geo) ──
with open(f"{DATA_PATH}/stations.json", "r") as f:
    stations_geo = json.load(f)
stations_list = []
for feat in stations_geo["features"]:
    props = feat["properties"]
    geom = feat.get("geometry")
    coords = geom["coordinates"] if geom and geom.get("coordinates") else [None, None]
    stations_list.append((
        props.get("code", ""), props.get("name", ""), props.get("state", ""),
        props.get("zone", ""), props.get("address", ""),
        coords[1] if coords[1] else None, coords[0] if coords[0] else None,
    ))
df_stations = spark.createDataFrame(
    stations_list,
    schema="station_code STRING, station_name STRING, state STRING, zone STRING, address STRING, latitude DOUBLE, longitude DOUBLE"
)
print(f"Stations: {df_stations.count()}")

# ── Load Trains (5,208 with class data) ──
with open(f"{DATA_PATH}/trains.json", "r") as f:
    trains_geo = json.load(f)
trains_list = []
for feat in trains_geo["features"]:
    props = feat["properties"]
    trains_list.append((
        str(props.get("number", "")), props.get("name", ""),
        props.get("from_station_code", ""), props.get("from_station_name", ""),
        props.get("to_station_code", ""), props.get("to_station_name", ""),
        props.get("zone", ""), props.get("type", ""),
        int(props.get("third_ac", 0) or 0), int(props.get("sleeper", 0) or 0),
        int(props.get("second_ac", 0) or 0), int(props.get("first_ac", 0) or 0),
        int(props.get("first_class", 0) or 0), int(props.get("chair_car", 0) or 0),
        props.get("departure", ""), props.get("arrival", ""),
        props.get("duration_h", 0), props.get("duration_m", 0),
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

# ── Build Router ──
class RailDrishtiRouter:
    def __init__(self):
        self.graph = defaultdict(list)
        self.hubs = set()
        self.station_names = {}

    def build_from_schedule(self, schedule_df):
        station_counts = (
            schedule_df.groupBy("station_code")
            .agg(F.countDistinct("train_number").alias("train_count"))
            .filter("train_count >= 10")
            .collect()
        )
        for row in station_counts:
            self.hubs.add(row.station_code)
        for row in schedule_df.select("station_code", "station_name").distinct().collect():
            if row.station_code and row.station_name:
                self.station_names[row.station_code] = row.station_name
        trains = defaultdict(list)
        for row in schedule_df.collect():
            trains[row.train_number].append(row)
        edge_count = 0
        for train_no, stops in trains.items():
            stops.sort(key=lambda x: (x.day or 0, x.id or 0))
            train_name = stops[0].train_name or train_no
            if len(stops) < 2:
                continue
            origin_stop, dest_stop = stops[0], stops[-1]
            for i, stop in enumerate(stops):
                if not stop.station_code:
                    continue
                if i > 0 and origin_stop.station_code and stop.station_code != origin_stop.station_code:
                    edge = {"train_no": train_no, "train_name": train_name, "from_stn": origin_stop.station_code, "to_stn": stop.station_code, "departure": (origin_stop.departure or "").replace(":00", "", 1) if origin_stop.departure else "", "arrival": (stop.arrival or "").replace(":00", "", 1) if stop.arrival else "", "day_from": origin_stop.day or 1, "day_to": stop.day or 1, "fare": 0, "fares": {}}
                    self.graph[origin_stop.station_code].append((stop.station_code, edge))
                    edge_count += 1
                if i < len(stops) - 1 and dest_stop.station_code and stop.station_code != dest_stop.station_code:
                    edge = {"train_no": train_no, "train_name": train_name, "from_stn": stop.station_code, "to_stn": dest_stop.station_code, "departure": (stop.departure or "").replace(":00", "", 1) if stop.departure else "", "arrival": (dest_stop.arrival or "").replace(":00", "", 1) if dest_stop.arrival else "", "day_from": stop.day or 1, "day_to": dest_stop.day or 1, "fare": 0, "fares": {}}
                    self.graph[stop.station_code].append((dest_stop.station_code, edge))
                    edge_count += 1
                if i < len(stops) - 1:
                    nxt = stops[i + 1]
                    if nxt.station_code and stop.station_code != nxt.station_code:
                        edge = {"train_no": train_no, "train_name": train_name, "from_stn": stop.station_code, "to_stn": nxt.station_code, "departure": (stop.departure or "").replace(":00", "", 1) if stop.departure else "", "arrival": (nxt.arrival or "").replace(":00", "", 1) if nxt.arrival else "", "day_from": stop.day or 1, "day_to": nxt.day or 1, "fare": 0, "fares": {}}
                        self.graph[stop.station_code].append((nxt.station_code, edge))
                        edge_count += 1
        print(f"Graph: {len(self.graph)} stations, {edge_count} edges, {len(self.hubs)} hubs, {len(trains)} trains")

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
        if travel_date is None: return None
        return travel_date + timedelta(minutes=mins)

    def _fmt_dt(self, dt):
        if dt is None: return ""
        return dt.strftime("%b %d %H:%M")

    def find_direct_routes(self, origin, dest, depart_after="00:00", travel_date=None):
        dep_min = self._time_to_mins(depart_after)
        routes, seen = [], set()
        for d, edge in self.graph.get(origin, []):
            if d == dest:
                dep = self._time_to_mins(edge["departure"], edge.get("day_from", 1))
                arr = self._time_to_mins(edge["arrival"], edge.get("day_to", 1))
                if dep >= dep_min:
                    travel = arr - dep
                    if travel <= 0: travel += 1440
                    key = edge["train_no"]
                    if key in seen: continue
                    seen.add(key)
                    dep_dt = self._mins_to_datetime(dep, travel_date)
                    arr_dt = self._mins_to_datetime(dep + travel, travel_date)
                    routes.append({"type": "DIRECT", "train_no": key, "train_name": edge["train_name"], "departure": edge["departure"], "arrival": edge["arrival"], "travel_time": self._format_mins(travel), "travel_mins": travel, "cheapest_fare": edge["fare"], "fares": edge["fares"], "availability": "NOT_CHECKED", "legs": 1, "via": "--", "depart_date": self._fmt_dt(dep_dt), "arrive_date": self._fmt_dt(arr_dt)})
        routes.sort(key=lambda r: r["travel_mins"])
        return routes

    def find_via_hub_routes(self, origin, dest, depart_after="00:00", max_hubs=10, min_conn=45, travel_date=None):
        dep_min = self._time_to_mins(depart_after)
        via_routes, seen = [], set()
        scored = [h for h in self.hubs if h != origin and h != dest and any(d == h for d, _ in self.graph.get(origin, [])) and any(d == dest for d, _ in self.graph.get(h, []))]
        for hub in scored[:max_hubs]:
            for _, l1 in self.graph.get(origin, []):
                if l1["to_stn"] != hub: continue
                l1d = self._time_to_mins(l1["departure"], l1.get("day_from", 1))
                if l1d < dep_min: continue
                l1a = self._time_to_mins(l1["arrival"], l1.get("day_to", 1))
                if l1a <= l1d: l1a += 1440
                for _, l2 in self.graph.get(hub, []):
                    if l2["to_stn"] != dest: continue
                    l2_dep_own = self._time_to_mins(l2["departure"], l2.get("day_from", 1))
                    l2_arr_own = self._time_to_mins(l2["arrival"], l2.get("day_to", 1))
                    if l2_arr_own <= l2_dep_own: l2_arr_own += 1440
                    leg2_duration = l2_arr_own - l2_dep_own
                    l2_dep_tod = self._time_to_mins(l2["departure"])
                    earliest_board = l1a + min_conn
                    l2d = l2_dep_tod
                    while l2d < earliest_board: l2d += 1440
                    conn_wait = l2d - l1a
                    if conn_wait > 2880: continue
                    l2a = l2d + leg2_duration
                    total = l2a - l1d
                    if total > 4320 or total < 0: continue
                    key = f"{l1['train_no']}_{l2['train_no']}_{hub}"
                    if key in seen: continue
                    seen.add(key)
                    dep_dt = self._mins_to_datetime(l1d, travel_date)
                    hub_arr_dt = self._mins_to_datetime(l1a, travel_date)
                    hub_dep_dt = self._mins_to_datetime(l2d, travel_date)
                    arr_dt = self._mins_to_datetime(l2a, travel_date)
                    via_routes.append({"type": f"VIA {hub}", "train_no": f"{l1['train_no']}+{l2['train_no']}", "train_name": f"{l1['train_name']} -> {l2['train_name']}", "departure": l1["departure"], "arrival": l2["arrival"], "travel_time": self._format_mins(total), "travel_mins": total, "cheapest_fare": 0, "fares": {}, "availability": "NOT_CHECKED", "legs": 2, "via": hub, "connection_wait": self._format_mins(conn_wait), "depart_date": self._fmt_dt(dep_dt), "hub_arrive_date": self._fmt_dt(hub_arr_dt), "hub_depart_date": self._fmt_dt(hub_dep_dt), "arrive_date": self._fmt_dt(arr_dt)})
        via_routes.sort(key=lambda r: r["travel_mins"])
        return via_routes

    def missed_train_protocol(self, origin, dest, current_time="00:00", travel_date=None, budget_max=None, max_hub_search=10):
        if travel_date is None:
            td = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif isinstance(travel_date, str):
            td = datetime.strptime(travel_date, "%Y-%m-%d")
        else:
            td = travel_date
        opts = []
        direct = self.find_direct_routes(origin, dest, depart_after=current_time, travel_date=td)
        opts.extend(direct)
        via = self.find_via_hub_routes(origin, dest, depart_after=current_time, max_hubs=max_hub_search, travel_date=td)
        opts.extend(via)
        opts.sort(key=lambda x: x["travel_mins"])
        return opts

router = RailDrishtiRouter()
router.build_from_schedule(df_schedule)
print("\n✅ Rail Drishti Engine loaded and ready!")

# COMMAND ----------

# DBTITLE 1,AI Functions: NLP Parser, Route Recommender, Waitlist Advisor
# ============================================================
# AI LAYER 1: Natural Language Travel Query Parser
# ============================================================
def ai_parse_travel_query(user_message: str) -> dict:
    prompt = f"""You are an Indian Railways travel assistant. Parse the user's travel query into structured JSON.
IMPORTANT: You must respond with ONLY valid JSON, no other text.
Extract these fields:
- origin_station_code: Railway station code for origin (e.g., NDLS for New Delhi, BCT for Mumbai Central)
- destination_station_code: Same for destination
- origin_city: City name
- destination_city: City name
- current_time: Time in HH:MM format (24hr). If 'evening' use 18:00, 'morning' 08:00, 'night' 22:00
- scenario: One of ['missed_train', 'planning_trip', 'looking_alternatives', 'checking_status', 'waitlist_query']
- train_mentioned: Train number if mentioned, else null
- travel_class_preference: One of ['SL', '3AC', '2AC', '1AC', 'any']
- budget_concern: true/false
- urgency: One of ['immediate', 'today', 'flexible']

Common station codes: Delhi: NDLS, NZM | Mumbai: BCT, CSTM, LTT | Kolkata: HWH | Chennai: MAS | Bangalore: SBC | Hyderabad: SC | Ahmedabad: ADI | Jaipur: JP | Lucknow: LKO | Varanasi: BSB | Bhopal: BPL | Pune: PUNE

User query: "{user_message}"
Respond with ONLY JSON:"""
    result = spark.sql(f"""
        SELECT ai_query('databricks-meta-llama-3-3-70b-instruct',
            "{prompt.replace('"', "'")}",
            modelParameters => named_struct('temperature', 0.1, 'max_tokens', 500)
        ) AS parsed
    """).collect()[0]["parsed"]
    try:
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
        parsed = json.loads(json_match.group()) if json_match else json.loads(result)
        return parsed
    except (json.JSONDecodeError, AttributeError) as e:
        return {"raw_response": result, "error": str(e)}


# ============================================================
# AI LAYER 2: Intelligent Route Recommender
# ============================================================
def ai_recommend_route(route_options: list, user_context: dict) -> str:
    routes_text = ""
    for i, r in enumerate(route_options[:10], 1):
        fare_str = f"Rs.{r['cheapest_fare']}" if r.get('cheapest_fare', 0) > 0 else "fare unknown"
        routes_text += (f"Route {i}: [{r['type']}] {r['train_name']} ({r['train_no']}) "
                       f"| Departs: {r.get('depart_date', r['departure'])} | Arrives: {r.get('arrive_date', r['arrival'])} "
                       f"| Time: {r['travel_time']} | Fare: {fare_str} "
                       f"| Legs: {r['legs']} | Via: {r.get('via', 'direct')}\n")
    origin = user_context.get("origin_city", "origin")
    dest = user_context.get("destination_city", "destination")
    scenario = user_context.get("scenario", "planning_trip")
    prompt = f"""You are Rail-Drishti, an expert Indian Railways travel advisor AI.
A passenger needs help from {origin} to {dest}. Scenario: {scenario}.
Routes:\n{routes_text}
Provide a clear recommendation in 4-6 sentences:
1. Which route you recommend FIRST and WHY
2. A backup option
3. Any warnings (tight connections, delays)
4. A money-saving tip if applicable
Speak like a knowledgeable Indian travel agent - practical, direct, helpful."""
    result = spark.sql(f"""
        SELECT ai_query('databricks-meta-llama-3-3-70b-instruct',
            "{prompt.replace('"', "'")}",
            modelParameters => named_struct('temperature', 0.5, 'max_tokens', 600)
        ) AS recommendation
    """).collect()[0]["recommendation"]
    return result


# ============================================================
# AI LAYER 3: Waitlist Advisor
# ============================================================
def ai_assess_waitlist(waitlist_position: int, train_type: str, travel_class: str, days_to_travel: int) -> str:
    prompt = f"""You are an Indian Railways waitlist prediction expert.
A passenger has WL/{waitlist_position} on a {train_type} train in {travel_class} class. Travel in {days_to_travel} days.
Estimate: 1. Confirmation probability (percentage) 2. Should they book or look for alternatives 3. Key factors affecting confirmation
Be specific. Keep to 3-4 sentences."""
    result = spark.sql(f"""
        SELECT ai_query('databricks-meta-llama-3-3-70b-instruct',
            "{prompt.replace('"', "'")}",
            modelParameters => named_struct('temperature', 0.3, 'max_tokens', 300)
        ) AS assessment
    """).collect()[0]["assessment"]
    return result


# ============================================================
# Station Lookup + Main AI Advisor
# ============================================================
def lookup_station_code(city_or_name: str) -> str:
    if not city_or_name: return None
    search = city_or_name.upper().strip()
    match = df_schedule.filter(F.upper(F.col("station_name")).contains(search)).select("station_code").first()
    if match: return match.station_code
    match = df_schedule.filter(F.upper(F.col("station_code")) == search).select("station_code").first()
    if match: return match.station_code
    match = df_stations.filter(F.upper(F.col("station_name")).contains(search) | F.upper(F.col("address")).contains(search)).first()
    if match:
        sched_match = df_schedule.filter(F.col("station_code") == match.station_code).first()
        if sched_match: return match.station_code
        sched_match = df_schedule.filter(F.upper(F.col("station_name")).contains(match.station_name.upper())).select("station_code").first()
        if sched_match: return sched_match.station_code
    return None


def rail_drishti_ai(user_message: str, travel_date: str = None) -> str:
    if travel_date is None:
        travel_dt = datetime.now()
        travel_date_str = travel_dt.strftime("%Y-%m-%d")
    else:
        travel_dt = datetime.strptime(travel_date, "%Y-%m-%d")
        travel_date_str = travel_date
    parsed = ai_parse_travel_query(user_message)
    if "error" in parsed:
        return "❌ Sorry, I couldn't understand your query. Try: 'Chennai to Ujjain' or 'Delhi to Mumbai trains'"
    scenario = parsed.get("scenario", "planning_trip")
    if scenario == "waitlist_query":
        wl_match = re.search(r'WL[/\-]?(\d+)', user_message, re.IGNORECASE)
        wl_pos = int(wl_match.group(1)) if wl_match else 45
        return ai_assess_waitlist(wl_pos, parsed.get("train_mentioned", "Express") or "Express", parsed.get("travel_class_preference", "SL") or "SL", 3)
    origin = parsed.get("origin_station_code")
    destination = parsed.get("destination_station_code")
    origin_city = parsed.get("origin_city", "")
    dest_city = parsed.get("destination_city", "")
    current_time = parsed.get("current_time", "08:00")
    if not origin or origin == "None":
        origin = lookup_station_code(origin_city)
        if not origin: return f"❌ Could not find station for '{origin_city}'."
    else:
        verify = df_schedule.filter(F.col("station_code") == origin).first()
        if not verify:
            resolved = lookup_station_code(origin_city)
            if resolved: origin = resolved
    if not destination or destination == "None":
        destination = lookup_station_code(dest_city)
        if not destination: return f"❌ Could not find station for '{dest_city}'."
    else:
        verify = df_schedule.filter(F.col("station_code") == destination).first()
        if not verify:
            resolved = lookup_station_code(dest_city)
            if resolved: destination = resolved
    route_options = router.missed_train_protocol(origin=origin, dest=destination, current_time=current_time, travel_date=travel_date_str, max_hub_search=10)
    if not route_options:
        return f"❌ No routes found from {origin} ({router.station_names.get(origin,'')}) to {destination} ({router.station_names.get(destination,'')}) after {current_time}."
    parsed["travel_date"] = travel_dt.strftime("%A %b %d, %Y")
    recommendation = ai_recommend_route(route_options, parsed)
    origin_name = router.station_names.get(origin, origin)
    dest_name = router.station_names.get(destination, destination)
    header = f"🚂 **{origin_name} → {dest_name}** | {len(route_options)} routes found\n\n"
    return header + recommendation

print("✅ AI Layers ready: NLP Parser, Route Recommender, Waitlist Advisor")

# COMMAND ----------

# DBTITLE 1,PDF Support, Chatbot Integration & Train Rating System
# ============================================================
# PDF SUPPORT
# ============================================================
pdf_contents = {}

def upload_pdf(file_path):
    global pdf_contents
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            filename = file_path.split('/')[-1]
            pdf_contents[filename] = text
            return f"✅ Loaded {filename} ({len(pdf_reader.pages)} pages)"
    except Exception as e:
        return f"❌ Error: {e}"

# Load PDFs
pdf_files = [
    "/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/SCR G&SR UPDATED AS16 HR.pdf",
    "/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/1463038915607-\u0938\u092e\u093e\u0928\u094d\u092f \u0928\u093f\u092f\u092e 2015 for  Printing.pdf",
    "/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/etktfaq.pdf",
    "/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/CitizenCharter.pdf",
    "/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/features_of_internet_tickets.pdf",
    "/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/tatkal_faq.pdf",
]
for f in pdf_files:
    try:
        print(upload_pdf(f))
    except:
        pass

BASE_SYSTEM_PROMPT = """You are a helpful train passenger assistant chatbot. You help passengers with:
- Train schedules and timings
- Platform information and directions
- Onboard facilities (WiFi, food, restrooms, charging points)
- Ticket booking and cancellation queries
- Baggage allowances and rules
- Safety and emergency procedures
- General travel tips and etiquette

Provide clear, concise, and friendly responses. Always respond in the SAME LANGUAGE as the user's question."""

def get_relevant_pdf_excerpt(query, max_chars=3000):
    if not pdf_contents: return None
    query_lower = query.lower()
    words = query_lower.split()
    all_scored = []
    for filename, content in pdf_contents.items():
        chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
        for chunk in chunks:
            score = sum(1 for word in words if len(word) > 3 and word in chunk.lower())
            if score > 0:
                all_scored.append((score, filename, chunk))
    if all_scored:
        all_scored.sort(reverse=True, key=lambda x: x[0])
        parts = []
        seen_pdfs = set()
        for score, filename, chunk in all_scored[:5]:
            if filename not in seen_pdfs:
                parts.append(f"\n--- FROM: {filename} ---")
                seen_pdfs.add(filename)
            parts.append(chunk[:800])
        return "\n\n".join(parts)[:max_chars]
    return None


# ============================================================
# INTEGRATED CHATBOT
# ============================================================
def is_travel_routing_query(message: str) -> bool:
    travel_keywords = [
        "train from", "train to", "trains from", "trains to",
        "route from", "route to", "travel from", "travel to",
        "missed train", "missed my train",
        "delhi to", "mumbai to", "chennai to", "kolkata to", "bangalore to",
        "hyderabad to", "pune to", "jaipur to", "lucknow to", "ahmedabad to",
        "varanasi to", "bhopal to", "kota to",
        "\u0938\u0947", "\u091c\u093e\u0928\u093e \u0939\u0948", "\u091f\u094d\u0930\u0947\u0928", "\u0917\u093e\u0921\u093c\u0940",
        "ndls", "bct", "hwh", "mas", "sbc", "cstm",
        "next train", "trains between", "direct train",
        "alternative train", "connecting train",
        "how to reach", "how to go",
        "waitlist", "wl/", "wl ", "waiting list", "confirmation chance",
    ]
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in travel_keywords)


def chat_with_assistant(message, history):
    try:
        if is_travel_routing_query(message):
            return rail_drishti_ai(message)
        pdf_excerpt = get_relevant_pdf_excerpt(message)
        if pdf_excerpt:
            pdf_list = ", ".join(pdf_contents.keys())
            system_prompt = f"""{BASE_SYSTEM_PROMPT}\n\nIMPORTANT: {len(pdf_contents)} PDF(s) loaded: {pdf_list}. Use this content:\n--- PDF CONTENT ---\n{pdf_excerpt}\n--- END ---"""
        else:
            system_prompt = BASE_SYSTEM_PROMPT
        messages = [ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt)]
        for user_msg, bot_msg in history:
            messages.append(ChatMessage(role=ChatMessageRole.USER, content=user_msg))
            messages.append(ChatMessage(role=ChatMessageRole.ASSISTANT, content=bot_msg))
        messages.append(ChatMessage(role=ChatMessageRole.USER, content=message))
        response = w.serving_endpoints.query(name="databricks-meta-llama-3-3-70b-instruct", messages=messages, temperature=0.7, max_tokens=500)
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}. Please try again."


# ============================================================
# TRAIN RATING SYSTEM (Persistent JSON Storage)
# ============================================================
def load_ratings():
    try:
        with open(RATINGS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_ratings(ratings):
    os.makedirs(os.path.dirname(RATINGS_FILE), exist_ok=True)
    with open(RATINGS_FILE, 'w') as f:
        json.dump(ratings, f, indent=2)

def add_rating(train_number, train_name, rating, comment, reviewer_name="Anonymous"):
    ratings = load_ratings()
    ratings.append({
        "train_number": train_number,
        "train_name": train_name,
        "rating": rating,
        "comment": comment,
        "reviewer": reviewer_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_ratings(ratings)
    return ratings

def get_ratings_summary():
    ratings = load_ratings()
    if not ratings:
        return {}
    summary = {}
    for r in ratings:
        key = f"{r['train_number']} - {r['train_name']}"
        if key not in summary:
            summary[key] = {"ratings": [], "comments": []}
        summary[key]["ratings"].append(r["rating"])
        summary[key]["comments"].append({"rating": r["rating"], "comment": r["comment"], "reviewer": r["reviewer"], "time": r["timestamp"]})
    for key in summary:
        avg = sum(summary[key]["ratings"]) / len(summary[key]["ratings"])
        summary[key]["avg_rating"] = round(avg, 1)
        summary[key]["count"] = len(summary[key]["ratings"])
    return summary

print("✅ PDF Knowledge Base loaded:", len(pdf_contents), "documents")
print("✅ Chatbot ready (travel routing + general Q&A)")
print("✅ Train Rating System ready (persistent JSON storage)")

# COMMAND ----------

# DBTITLE 1,Rail Drishti Dashboard - Beautiful Gradio UI
# ============================================================
# RAIL DRISHTI DASHBOARD - Complete Gradio UI
# ============================================================

# Compute stats for the dashboard
total_trains = df_trains.count()
total_stations = df_stations.count()
total_schedules = df_schedule.count()
total_hubs = len(router.hubs)
total_zones = df_trains.select("zone").distinct().count()
total_pdfs = len(pdf_contents)

# Zone distribution
zone_data = df_trains.groupBy("zone").count().orderBy(F.desc("count")).collect()
zone_stats = "\n".join([f"  {r.zone or 'Unknown'}: {r['count']} trains" for r in zone_data[:10]])

# Class availability
class_stats = df_trains.agg(
    F.sum(F.when(F.col("sleeper") > 0, 1).otherwise(0)).alias("sleeper"),
    F.sum(F.when(F.col("third_ac") > 0, 1).otherwise(0)).alias("third_ac"),
    F.sum(F.when(F.col("second_ac") > 0, 1).otherwise(0)).alias("second_ac"),
    F.sum(F.when(F.col("first_ac") > 0, 1).otherwise(0)).alias("first_ac"),
    F.sum(F.when(F.col("chair_car") > 0, 1).otherwise(0)).alias("chair_car"),
).collect()[0]

# Top connected stations
top_stations = (
    df_schedule.groupBy("station_code", "station_name")
    .agg(F.countDistinct("train_number").alias("trains"))
    .orderBy(F.desc("trains"))
    .limit(15)
    .collect()
)

# State distribution
state_data = df_stations.filter(F.col("state").isNotNull()).groupBy("state").count().orderBy(F.desc("count")).limit(10).collect()

# Custom CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', sans-serif !important;
    max-width: 1400px !important;
}

/* Header styling */
.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 30px 40px;
    border-radius: 16px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(233,196,106,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.header-banner h1 {
    font-size: 2.2em;
    font-weight: 700;
    margin: 0;
    color: #e9c46a;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}
.header-banner p {
    font-size: 1.1em;
    opacity: 0.9;
    margin-top: 8px;
    color: #a8dadc;
}

/* Stat cards */
.stat-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 24px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-3px); }
.stat-card h3 { font-size: 2.5em; margin: 0; font-weight: 700; }
.stat-card p { font-size: 0.9em; opacity: 0.9; margin-top: 4px; }

.stat-card-orange {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 16px; padding: 24px; color: white; text-align: center;
    box-shadow: 0 4px 15px rgba(245,87,108,0.4); transition: transform 0.2s;
}
.stat-card-orange:hover { transform: translateY(-3px); }
.stat-card-orange h3 { font-size: 2.5em; margin: 0; font-weight: 700; }
.stat-card-orange p { font-size: 0.9em; opacity: 0.9; margin-top: 4px; }

.stat-card-green {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    border-radius: 16px; padding: 24px; color: #1a1a2e; text-align: center;
    box-shadow: 0 4px 15px rgba(67,233,123,0.4); transition: transform 0.2s;
}
.stat-card-green:hover { transform: translateY(-3px); }
.stat-card-green h3 { font-size: 2.5em; margin: 0; font-weight: 700; }
.stat-card-green p { font-size: 0.9em; opacity: 0.9; margin-top: 4px; }

.stat-card-gold {
    background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    border-radius: 16px; padding: 24px; color: #1a1a2e; text-align: center;
    box-shadow: 0 4px 15px rgba(247,151,30,0.4); transition: transform 0.2s;
}
.stat-card-gold:hover { transform: translateY(-3px); }
.stat-card-gold h3 { font-size: 2.5em; margin: 0; font-weight: 700; }
.stat-card-gold p { font-size: 0.9em; opacity: 0.9; margin-top: 4px; }

/* Tab styling */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 1em !important;
    padding: 12px 20px !important;
    border-radius: 8px 8px 0 0 !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* Info box */
.info-box {
    background: linear-gradient(135deg, #e8f4f8 0%, #d4e6f1 100%);
    border-left: 4px solid #2980b9;
    padding: 16px 20px;
    border-radius: 0 12px 12px 0;
    margin: 10px 0;
}

/* Rating stars */
.rating-display {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    padding: 20px;
    border-radius: 12px;
    margin: 8px 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* Route result card */
.route-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Chatbot styling */
.chatbot-container .message { border-radius: 12px !important; }
"""

# ============================================================
# GRADIO APP BUILDER
# ============================================================

def build_dashboard():
    with gr.Blocks(css=custom_css, title="Rail Drishti - AI Railway Intelligence", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple")) as app:

        # ---- HEADER ----
        gr.HTML(f"""
        <div class="header-banner">
            <h1>\U0001f682 Rail-Drishti</h1>
            <p>AI-Powered Indian Railways Intelligence Platform | {total_trains:,} Trains \u2022 {total_stations:,} Stations \u2022 {total_schedules:,} Schedule Entries</p>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ================================================================
            # TAB 1: HOME / OVERVIEW
            # ================================================================
            with gr.Tab("\U0001f3e0 Home", id="home"):
                gr.Markdown("### \U0001f4ca Network Overview")
                with gr.Row():
                    gr.HTML(f'<div class="stat-card"><h3>{total_trains:,}</h3><p>\U0001f682 Total Trains</p></div>')
                    gr.HTML(f'<div class="stat-card-orange"><h3>{total_stations:,}</h3><p>\U0001f6e4\ufe0f Stations</p></div>')
                    gr.HTML(f'<div class="stat-card-green"><h3>{total_hubs:,}</h3><p>\U0001f310 Hub Stations</p></div>')
                    gr.HTML(f'<div class="stat-card-gold"><h3>{total_pdfs}</h3><p>\U0001f4da PDF Documents</p></div>')

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### \U0001f3c6 Top Connected Stations")
                        top_stn_html = '<table style="width:100%;border-collapse:collapse;">'
                        top_stn_html += '<tr style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;"><th style="padding:10px;border-radius:8px 0 0 0;">Rank</th><th style="padding:10px;">Station</th><th style="padding:10px;">Code</th><th style="padding:10px;border-radius:0 8px 0 0;">Trains</th></tr>'
                        for i, s in enumerate(top_stations, 1):
                            bg = '#f8f9fa' if i % 2 == 0 else 'white'
                            medal = ['\U0001f947','\U0001f948','\U0001f949'][i-1] if i <= 3 else f'{i}.'
                            top_stn_html += f'<tr style="background:{bg};"><td style="padding:8px 10px;text-align:center;">{medal}</td><td style="padding:8px 10px;font-weight:500;">{s.station_name}</td><td style="padding:8px 10px;text-align:center;"><code>{s.station_code}</code></td><td style="padding:8px 10px;text-align:center;font-weight:600;color:#667eea;">{s.trains}</td></tr>'
                        top_stn_html += '</table>'
                        gr.HTML(top_stn_html)

                    with gr.Column(scale=1):
                        gr.Markdown("### \U0001f683 Class Availability")
                        class_html = '<div style="display:grid;gap:8px;">'
                        classes = [
                            ("Sleeper (SL)", class_stats.sleeper, "#43e97b"),
                            ("AC 3-Tier (3AC)", class_stats.third_ac, "#667eea"),
                            ("AC 2-Tier (2AC)", class_stats.second_ac, "#f5576c"),
                            ("AC 1st Class (1AC)", class_stats.first_ac, "#f7971e"),
                            ("Chair Car (CC)", class_stats.chair_car, "#a855f7"),
                        ]
                        for name, count, color in classes:
                            pct = (count / total_trains * 100) if total_trains else 0
                            class_html += f'<div style="background:#f8f9fa;border-radius:8px;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;"><span style="font-weight:500;">{name}</span><div style="display:flex;align-items:center;gap:10px;"><div style="width:120px;height:8px;background:#e0e0e0;border-radius:4px;overflow:hidden;"><div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div></div><span style="font-weight:600;color:{color};min-width:50px;text-align:right;">{count:,}</span></div></div>'
                        class_html += '</div>'
                        gr.HTML(class_html)

                        gr.Markdown("### \U0001f5fa\ufe0f Zone Distribution (Top 10)")
                        zone_html = '<div style="display:grid;gap:6px;">'
                        max_zone = zone_data[0]['count'] if zone_data else 1
                        for r in zone_data[:10]:
                            pct = (r['count'] / max_zone * 100)
                            zone_html += f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;"><span style="min-width:60px;font-weight:500;">{r.zone or "N/A"}</span><div style="flex:1;height:22px;background:#e8eaf6;border-radius:6px;overflow:hidden;"><div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#667eea,#764ba2);border-radius:6px;display:flex;align-items:center;padding-left:8px;"><span style="color:white;font-size:0.8em;font-weight:600;">{r["count"]}</span></div></div></div>'
                        zone_html += '</div>'
                        gr.HTML(zone_html)

            # ================================================================
            # TAB 2: AI TRAVEL ASSISTANT (CHATBOT)
            # ================================================================
            with gr.Tab("\U0001f916 AI Assistant", id="assistant"):
                gr.Markdown("### \U0001f916 Rail-Drishti AI Travel Assistant")
                gr.HTML('<div class="info-box"><strong>\U0001f4a1 I can help with:</strong> Route planning, missed trains, waitlist predictions, ticket rules, baggage info, and more!<br><strong>Examples:</strong> "Delhi to Mumbai trains" \u2022 "I missed my Rajdhani" \u2022 "What is baggage allowance?" \u2022 "WL/45 on Rajdhani 3AC"</div>')

                chatbot = gr.Chatbot(height=480, show_label=False, elem_classes=["chatbot-container"], avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/locomotive_1f682.png"))
                with gr.Row():
                    chat_input = gr.Textbox(placeholder="Ask me anything about Indian Railways...", show_label=False, scale=6, container=False)
                    send_btn = gr.Button("Send \U0001f680", variant="primary", scale=1)
                clear_btn = gr.Button("\U0001f5d1\ufe0f Clear Chat", variant="secondary", size="sm")

                def respond(message, history):
                    if not message.strip():
                        return history, ""
                    history = history or []
                    bot_response = chat_with_assistant(message, history)
                    history.append((message, bot_response))
                    return history, ""

                send_btn.click(respond, [chat_input, chatbot], [chatbot, chat_input])
                chat_input.submit(respond, [chat_input, chatbot], [chatbot, chat_input])
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_input])

            # ================================================================
            # TAB 3: ROUTE PLANNER
            # ================================================================
            with gr.Tab("\U0001f5fa\ufe0f Route Planner", id="routes"):
                gr.Markdown("### \U0001f5fa\ufe0f Smart Route Planner")
                gr.HTML('<div class="info-box">Enter origin and destination to find the best routes. Supports station codes (NDLS, BCT) or city names (Delhi, Mumbai).</div>')

                with gr.Row():
                    origin_input = gr.Textbox(label="\U0001f7e2 Origin Station", placeholder="e.g., Delhi, NDLS, Chennai", scale=2)
                    dest_input = gr.Textbox(label="\U0001f534 Destination Station", placeholder="e.g., Mumbai, BCT, Kolkata", scale=2)
                    time_input = gr.Textbox(label="\u23f0 After Time", placeholder="HH:MM", value="08:00", scale=1)
                    date_input = gr.Textbox(label="\U0001f4c5 Date", placeholder="YYYY-MM-DD", value=datetime.now().strftime("%Y-%m-%d"), scale=1)

                search_btn = gr.Button("\U0001f50d Find Routes", variant="primary", size="lg")
                route_output = gr.HTML(label="Route Results")

                def search_routes(origin, dest, time, date):
                    if not origin or not dest:
                        return '<div class="info-box">\u26a0\ufe0f Please enter both origin and destination.</div>'
                    origin_code = lookup_station_code(origin) or origin.upper().strip()
                    dest_code = lookup_station_code(dest) or dest.upper().strip()
                    origin_name = router.station_names.get(origin_code, origin)
                    dest_name = router.station_names.get(dest_code, dest)
                    try:
                        routes = router.missed_train_protocol(origin=origin_code, dest=dest_code, current_time=time or "08:00", travel_date=date, max_hub_search=10)
                    except Exception as e:
                        return f'<div class="info-box">\u274c Error: {e}</div>'
                    if not routes:
                        return f'<div class="info-box">\u274c No routes found from {origin_name} ({origin_code}) to {dest_name} ({dest_code}) after {time}.</div>'
                    html = f'<div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white;padding:20px;border-radius:12px;margin-bottom:15px;"><h3 style="margin:0;color:#e9c46a;">\U0001f682 {origin_name} \u2192 {dest_name}</h3><p style="margin:5px 0 0;opacity:0.8;">{len(routes)} routes found | Date: {date} | After: {time}</p></div>'
                    html += '<table style="width:100%;border-collapse:separate;border-spacing:0 6px;">'
                    html += '<tr style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;"><th style="padding:10px;border-radius:8px 0 0 8px;">#</th><th style="padding:10px;">Type</th><th style="padding:10px;">Train</th><th style="padding:10px;">Departs</th><th style="padding:10px;">Arrives</th><th style="padding:10px;">Duration</th><th style="padding:10px;border-radius:0 8px 8px 0;">Via</th></tr>'
                    for i, r in enumerate(routes[:15], 1):
                        bg = '#f0f4ff' if i % 2 == 0 else 'white'
                        type_color = '#43e97b' if r['type'] == 'DIRECT' else '#f5576c'
                        type_badge = f'<span style="background:{type_color};color:white;padding:2px 8px;border-radius:12px;font-size:0.8em;font-weight:600;">{r["type"]}</span>'
                        html += f'<tr style="background:{bg};"><td style="padding:10px;text-align:center;font-weight:600;">{i}</td><td style="padding:10px;">{type_badge}</td><td style="padding:10px;"><strong>{r["train_name"]}</strong><br><span style="color:#888;font-size:0.85em;">{r["train_no"]}</span></td><td style="padding:10px;font-weight:500;">{r.get("depart_date", r["departure"])}</td><td style="padding:10px;font-weight:500;">{r.get("arrive_date", r["arrival"])}</td><td style="padding:10px;text-align:center;"><span style="background:#e8eaf6;padding:4px 10px;border-radius:8px;font-weight:600;color:#667eea;">{r["travel_time"]}</span></td><td style="padding:10px;text-align:center;">{r.get("via","--")}</td></tr>'
                    html += '</table>'
                    return html

                search_btn.click(search_routes, [origin_input, dest_input, time_input, date_input], route_output)

            # ================================================================
            # TAB 4: TRAIN RATINGS
            # ================================================================
            with gr.Tab("\u2b50 Train Ratings", id="ratings"):
                gr.Markdown("### \u2b50 Train Rating & Review System")
                gr.HTML('<div class="info-box">Rate your train experience! Your ratings are stored persistently and help other passengers make informed decisions.</div>')

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### \U0001f4dd Submit a Rating")
                        rating_train_no = gr.Textbox(label="Train Number", placeholder="e.g., 12952")
                        rating_train_name = gr.Textbox(label="Train Name", placeholder="e.g., Mumbai Rajdhani")
                        rating_value = gr.Slider(label="Rating", minimum=1, maximum=5, step=1, value=4, info="1=Poor, 5=Excellent")
                        rating_comment = gr.Textbox(label="Your Review", placeholder="Share your experience...", lines=3)
                        rating_reviewer = gr.Textbox(label="Your Name (optional)", placeholder="Anonymous", value="Anonymous")
                        submit_rating_btn = gr.Button("\u2b50 Submit Rating", variant="primary")
                        rating_status = gr.HTML()

                    with gr.Column(scale=1):
                        gr.Markdown("#### \U0001f4ca All Ratings & Reviews")
                        ratings_display = gr.HTML()
                        refresh_ratings_btn = gr.Button("\U0001f504 Refresh Ratings", variant="secondary")

                def submit_rating(train_no, train_name, rating, comment, reviewer):
                    if not train_no:
                        return '<div style="color:#f5576c;padding:10px;">\u26a0\ufe0f Please enter a train number.</div>', display_ratings()
                    if not train_name:
                        match = df_trains.filter(F.col("train_number") == train_no).select("train_name").first()
                        train_name = match.train_name if match else f"Train {train_no}"
                    add_rating(train_no, train_name, rating, comment or "No comment", reviewer or "Anonymous")
                    stars = '\u2b50' * int(rating)
                    status = f'<div style="background:linear-gradient(135deg,#43e97b,#38f9d7);padding:15px;border-radius:12px;color:#1a1a2e;"><strong>\u2705 Rating submitted!</strong><br>{stars} for {train_name} ({train_no})</div>'
                    return status, display_ratings()

                def display_ratings():
                    summary = get_ratings_summary()
                    if not summary:
                        return '<div style="text-align:center;padding:40px;color:#888;"><h3>\U0001f4ad No ratings yet</h3><p>Be the first to rate a train!</p></div>'
                    html = ''
                    for train, data in sorted(summary.items(), key=lambda x: x[1]['avg_rating'], reverse=True):
                        stars = '\u2b50' * round(data['avg_rating'])
                        empty = '\u2606' * (5 - round(data['avg_rating']))
                        html += f'<div class="rating-display"><div style="display:flex;justify-content:space-between;align-items:center;"><strong style="font-size:1.1em;">{train}</strong><span style="font-size:1.3em;">{stars}{empty} <strong>{data["avg_rating"]}</strong>/5</span></div><div style="color:#666;font-size:0.9em;margin-top:4px;">{data["count"]} review(s)</div>'
                        for c in data['comments'][-3:]:
                            c_stars = '\u2b50' * int(c['rating'])
                            html += f'<div style="background:white;border-radius:8px;padding:10px 14px;margin-top:8px;border-left:3px solid #f7971e;"><div style="display:flex;justify-content:space-between;"><span style="font-weight:500;">{c["reviewer"]}</span><span>{c_stars}</span></div><p style="margin:4px 0 0;color:#555;">{c["comment"]}</p><span style="font-size:0.75em;color:#999;">{c["time"]}</span></div>'
                        html += '</div>'
                    return html

                submit_rating_btn.click(submit_rating, [rating_train_no, rating_train_name, rating_value, rating_comment, rating_reviewer], [rating_status, ratings_display])
                refresh_ratings_btn.click(display_ratings, outputs=ratings_display)
                app.load(display_ratings, outputs=ratings_display)

            # ================================================================
            # TAB 5: ANALYTICS
            # ================================================================
            with gr.Tab("\U0001f4ca Analytics", id="analytics"):
                gr.Markdown("### \U0001f4ca Indian Railways Analytics")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### \U0001f5fa\ufe0f Stations by State (Top 10)")
                        state_html = '<table style="width:100%;border-collapse:collapse;"><tr style="background:linear-gradient(135deg,#f093fb,#f5576c);color:white;"><th style="padding:10px;">State</th><th style="padding:10px;">Stations</th><th style="padding:10px;">Distribution</th></tr>'
                        max_state = state_data[0]['count'] if state_data else 1
                        for i, s in enumerate(state_data):
                            bg = '#fff5f5' if i % 2 == 0 else 'white'
                            pct = s['count'] / max_state * 100
                            state_html += f'<tr style="background:{bg};"><td style="padding:8px 10px;font-weight:500;">{s.state}</td><td style="padding:8px 10px;text-align:center;font-weight:600;">{s["count"]}</td><td style="padding:8px 10px;"><div style="width:100%;background:#f0f0f0;border-radius:4px;overflow:hidden;height:16px;"><div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#f093fb,#f5576c);border-radius:4px;"></div></div></td></tr>'
                        state_html += '</table>'
                        gr.HTML(state_html)

                    with gr.Column():
                        gr.Markdown("#### \U0001f50d Train Search")
                        train_search_input = gr.Textbox(label="Search by train number or name", placeholder="e.g., 12952 or Rajdhani")
                        train_search_btn = gr.Button("\U0001f50d Search", variant="primary")
                        train_search_output = gr.HTML()

                        def search_train(query):
                            if not query: return '<div class="info-box">Enter a train number or name to search.</div>'
                            results = df_trains.filter(
                                F.col("train_number").contains(query.strip()) |
                                F.upper(F.col("train_name")).contains(query.upper().strip())
                            ).limit(10).collect()
                            if not results:
                                return f'<div class="info-box">\u274c No trains found for "{query}".</div>'
                            html = '<div style="display:grid;gap:8px;">'
                            for t in results:
                                classes = []
                                if t.sleeper: classes.append(f'SL({t.sleeper})')
                                if t.third_ac: classes.append(f'3AC({t.third_ac})')
                                if t.second_ac: classes.append(f'2AC({t.second_ac})')
                                if t.first_ac: classes.append(f'1AC({t.first_ac})')
                                if t.chair_car: classes.append(f'CC({t.chair_car})')
                                cls_str = ' \u2022 '.join(classes) if classes else 'N/A'
                                dur = f"{t.duration_h}h {t.duration_m}m" if t.duration_h else 'N/A'
                                html += f'<div style="background:white;border:1px solid #e0e0e0;border-radius:12px;padding:14px;box-shadow:0 2px 6px rgba(0,0,0,0.05);"><div style="display:flex;justify-content:space-between;align-items:center;"><strong style="color:#667eea;font-size:1.1em;">{t.train_name}</strong><code style="background:#e8eaf6;padding:4px 10px;border-radius:6px;">{t.train_number}</code></div><div style="margin-top:6px;color:#555;">\U0001f7e2 {t.from_station_name} ({t.from_station_code}) \u2192 \U0001f534 {t.to_station_name} ({t.to_station_code})</div><div style="margin-top:4px;font-size:0.9em;"><span style="color:#888;">Duration:</span> <strong>{dur}</strong> \u2022 <span style="color:#888;">Zone:</span> <strong>{t.zone}</strong></div><div style="margin-top:4px;font-size:0.85em;color:#764ba2;">Classes: {cls_str}</div></div>'
                            html += '</div>'
                            return html

                        train_search_btn.click(search_train, train_search_input, train_search_output)
                        train_search_input.submit(search_train, train_search_input, train_search_output)

            # ================================================================
            # TAB 6: PASSENGER GUIDE
            # ================================================================
            with gr.Tab("\U0001f4da Passenger Guide", id="guide"):
                gr.Markdown("### \U0001f4da Passenger Information Guide")
                _pdf_names = " \u2022 ".join(pdf_contents.keys()) if pdf_contents else "None"
                gr.HTML(f'<div class="info-box"><strong>\U0001f4d6 {len(pdf_contents)} documents loaded:</strong> {_pdf_names}</div>')

                with gr.Row():
                    with gr.Column(scale=2):
                        guide_question = gr.Textbox(label="Ask a question about railway rules, tickets, or policies", placeholder="e.g., What is the baggage allowance for AC 2-Tier?")
                        guide_btn = gr.Button("\U0001f4d6 Find Answer", variant="primary")
                        guide_output = gr.Markdown()

                        def answer_guide_question(question):
                            if not question: return "\u26a0\ufe0f Please enter a question."
                            return chat_with_assistant(question, [])

                        guide_btn.click(answer_guide_question, guide_question, guide_output)
                        guide_question.submit(answer_guide_question, guide_question, guide_output)

                    with gr.Column(scale=1):
                        gr.Markdown("#### \u2753 Quick Reference")
                        quick_topics = [
                            ("\U0001f4bc Baggage Allowance", "What is the baggage allowance for different classes?"),
                            ("\U0001f3ab Ticket Cancellation", "What is the ticket cancellation and refund policy?"),
                            ("\u23f0 Tatkal Booking", "What are the tatkal booking timings and rules?"),
                            ("\U0001f6c8 Onboard Facilities", "What facilities are available on trains?"),
                            ("\u26a0\ufe0f Emergency Procedures", "What should I do in case of emergency on train?"),
                            ("\U0001f4f1 E-Ticket Rules", "What are the rules for e-tickets and i-tickets?"),
                        ]
                        for topic_name, topic_q in quick_topics:
                            gr.Button(topic_name, size="sm").click(lambda q=topic_q: q, outputs=guide_question)

        # ---- FOOTER ----
        gr.HTML("""
        <div style="text-align:center;padding:20px;margin-top:20px;border-top:1px solid #e0e0e0;color:#888;">
            <p>\U0001f6e4\ufe0f <strong>Rail-Drishti</strong> | Built with Databricks AI + Gradio | Data: Datameet Indian Railways Open Dataset</p>
            <p style="font-size:0.85em;">5,208 Trains \u2022 8,990 Stations \u2022 417,080 Schedule Entries \u2022 AI-Powered by Meta Llama 3.3 70B</p>
        </div>
        """)

    return app

print("\u2705 Dashboard builder ready! Run the next cell to launch.")

# COMMAND ----------

# DBTITLE 1,Launch Rail Drishti Dashboard
# ============================================================
# LAUNCH THE DASHBOARD
# ============================================================
import gradio as gr

# Close any stale Gradio servers first
try:
    gr.close_all()
except:
    pass

app = build_dashboard()
app.launch()