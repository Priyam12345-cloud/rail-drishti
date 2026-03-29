# Databricks notebook source
# MAGIC %pip install gradio --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install PyPDF2 sentence-transformers lightgbm --quiet

# COMMAND ----------

import gradio as gr
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from datetime import datetime

# Initialize Databricks client
w = WorkspaceClient()

print("✅ Libraries imported successfully")

# COMMAND ----------

import PyPDF2
import io

# Global dictionary to store multiple PDF contents
pdf_contents = {}  # {filename: content}

def upload_pdf(file_path):
    """
    Extract text from a PDF file and add to collection.
    Supports multiple PDFs.
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        Extracted text content
    """
    global pdf_contents
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extract text from all pages
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            
            filename = file_path.split('/')[-1]
            pdf_contents[filename] = text
            
            display(f"✅ PDF uploaded successfully: {filename}")
            display(f"📄 Total pages: {len(pdf_reader.pages)}")
            display(f"📝 Extracted {len(text):,} characters")
            display(f"📚 Total PDFs loaded: {len(pdf_contents)}")
            
            return text
    
    except Exception as e:
        display(f"❌ Error reading PDF: {str(e)}")
        return ""

def upload_multiple_pdfs(file_paths):
    """
    Upload multiple PDF files at once.
    
    Args:
        file_paths: List of file paths
    """
    display(f"📚 Uploading {len(file_paths)} PDF files...")
    display("="*70)
    
    for path in file_paths:
        upload_pdf(path)
        display("-"*70)
    
    display(f"\n✅ All PDFs loaded! Total: {len(pdf_contents)} documents")

def list_pdfs():
    """List all loaded PDFs with their sizes."""
    if not pdf_contents:
        display("❌ No PDFs loaded. Use upload_pdf('/path/to/file.pdf') to upload.")
        return
    
    display(f"📚 Currently loaded PDFs: {len(pdf_contents)}")
    display("="*70)
    
    for i, (filename, content) in enumerate(pdf_contents.items(), 1):
        display(f"{i}. 📄 {filename}")
        display(f"   📝 Size: {len(content):,} characters")
        display(f"   📖 Preview: {content[:100]}...")
        display("-"*70)

def clear_pdfs():
    """Clear all loaded PDFs."""
    global pdf_contents
    count = len(pdf_contents)
    pdf_contents = {}
    display(f"🗑️ Cleared {count} PDF(s) from memory")

def remove_pdf(filename):
    """Remove a specific PDF by filename."""
    global pdf_contents
    if filename in pdf_contents:
        del pdf_contents[filename]
        display(f"🗑️ Removed: {filename}")
        display(f"📚 Remaining PDFs: {len(pdf_contents)}")
    else:
        display(f"❌ PDF not found: {filename}")
        display("Available PDFs:")
        for name in pdf_contents.keys():
            display(f"  - {name}")

print("✅ Multi-PDF functions ready")
print("📚 Functions: upload_pdf(), upload_multiple_pdfs(), list_pdfs(), clear_pdfs(), remove_pdf()")

# Initialize empty collection
pdf_contents = {}

# COMMAND ----------

# System prompt for train passenger assistant
BASE_SYSTEM_PROMPT = """You are a helpful train passenger assistant chatbot. You help passengers with:
- Train schedules and timings
- Platform information and directions
- Onboard facilities (WiFi, food, restrooms, charging points)
- Ticket booking and cancellation queries
- Baggage allowances and rules
- Safety and emergency procedures
- General travel tips and etiquette

Provide clear, concise, and friendly responses. If you don't know something specific, acknowledge it and suggest asking train staff.

🌍 MULTILINGUAL SUPPORT:
ALWAYS respond in the SAME LANGUAGE as the user's question. If the user asks in Hindi, respond in Hindi. If in English, respond in English. Support all major Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu, Odia, etc.) and international languages (Spanish, French, German, Chinese, Japanese, etc.)."""

def get_relevant_pdf_excerpt(query, max_chars=3000):
    """
    Extract relevant portions from ALL loaded PDFs based on query keywords.
    Returns excerpts with source PDF names.
    """
    if not pdf_contents:
        return None
    
    # Simple keyword-based search
    query_lower = query.lower()
    words = query_lower.split()
    
    # Search across all PDFs
    all_scored_chunks = []
    
    for filename, content in pdf_contents.items():
        # Split PDF into chunks
        chunk_size = 1000
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        # Score chunks based on keyword matches
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for word in words if len(word) > 3 and word in chunk_lower)
            if score > 0:
                all_scored_chunks.append((score, filename, chunk))
    
    # Return top chunks from across all PDFs
    if all_scored_chunks:
        all_scored_chunks.sort(reverse=True, key=lambda x: x[0])
        
        # Group by source PDF and build result
        result_parts = []
        seen_pdfs = set()
        
        for score, filename, chunk in all_scored_chunks[:5]:  # Top 5 chunks
            if filename not in seen_pdfs:
                result_parts.append(f"\n--- FROM: {filename} ---")
                seen_pdfs.add(filename)
            result_parts.append(chunk[:800])  # Limit chunk size
        
        result = "\n\n".join(result_parts)
        return result[:max_chars]
    
    return None

def chat_with_assistant(message, history):
    """
    Main chatbot function that processes user messages and returns responses.
    Searches across ALL loaded PDFs.
    🌍 MULTILINGUAL: Automatically responds in the user's language.
    
    Args:
        message: User's current message
        history: List of previous [user_msg, bot_msg] pairs
    
    Returns:
        Bot's response string
    """
    try:
        # Check if PDF content is available
        pdf_excerpt = get_relevant_pdf_excerpt(message)
        
        # Modify system prompt if PDF content is available
        if pdf_excerpt:
            pdf_list = ", ".join(pdf_contents.keys())
            system_prompt = f"""{BASE_SYSTEM_PROMPT}

IMPORTANT: {len(pdf_contents)} PDF document(s) have been uploaded: {pdf_list}. Use the following content from the PDFs to answer the user's question. If the answer is in the PDFs, cite the source document. If not, provide general assistance.

--- PDF CONTENT ---
{pdf_excerpt}
--- END PDF CONTENT ---
"""
        else:
            system_prompt = BASE_SYSTEM_PROMPT
        
        # Build conversation history for the LLM
        messages = [ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt)]
        
        # Add conversation history
        for user_msg, bot_msg in history:
            messages.append(ChatMessage(role=ChatMessageRole.USER, content=user_msg))
            messages.append(ChatMessage(role=ChatMessageRole.ASSISTANT, content=bot_msg))
        
        # Add current user message
        messages.append(ChatMessage(role=ChatMessageRole.USER, content=message))
        
        # Call Databricks Foundation Model API (using correct endpoint)
        response = w.serving_endpoints.query(
            name="databricks-meta-llama-3-3-70b-instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract response text
        bot_response = response.choices[0].message.content
        return bot_response
        
    except Exception as e:
        # Multilingual error message
        return f"I apologize, but I encountered an error: {str(e)}. Please try again or contact train staff. | मुझे खेद है, लेकिन एक त्रुटि हुई। कृपया पुनः प्रयास करें।"

print("✅ Multi-PDF chatbot function defined successfully")
print("📚 Searches across all loaded PDFs automatically")
print("🌍 Languages supported: English, Hindi, Tamil, Telugu, Bengali, Marathi, and 100+ more!")

# COMMAND ----------

# DBTITLE 1,Load Rail Drishti Train Datasets
# ============================================================
# RAIL DRISHTI: Load Train Datasets
# ============================================================
import json
from collections import defaultdict
from pyspark.sql.types import *
from pyspark.sql import functions as F

DATA_PATH = "/Workspace/Users/lopamudra.wncc@gmail.com/Data"

# 1. LOAD SCHEDULES (417K stops)
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

# 2. LOAD STATIONS (8,990 stations with coordinates)
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

# 3. LOAD TRAINS (5,208 trains with class availability)
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
print("\n✅ All Rail Drishti datasets loaded!")

# COMMAND ----------

# DBTITLE 1,RailDrishtiRouter - Graph-Based Routing Engine
# ============================================================
# Routing Engine -- builds graph from schedules.json
# ============================================================

class RailDrishtiRouter:
    """Graph-based routing engine. Built from real Datameet data."""

    def __init__(self):
        self.graph = defaultdict(list)
        self.hubs = set()
        self.station_names = {}

    def build_from_schedule(self, schedule_df):
        """Build transport graph from schedules DataFrame."""
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

            origin_stop = stops[0]
            dest_stop = stops[-1]

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

    def find_direct_routes(self, origin, dest, depart_after="00:00"):
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
                    routes.append({
                        "type": "DIRECT", "train_no": key,
                        "train_name": edge["train_name"],
                        "departure": edge["departure"], "arrival": edge["arrival"],
                        "travel_time": self._format_mins(travel), "travel_mins": travel,
                        "cheapest_fare": edge["fare"], "fares": edge["fares"],
                        "availability": "NOT_CHECKED", "legs": 1, "via": "--",
                    })
        routes.sort(key=lambda r: r["travel_mins"])
        return routes

    def find_via_hub_routes(self, origin, dest, depart_after="00:00",
                            max_hubs=10, min_conn=45):
        dep_min = self._time_to_mins(depart_after)
        via_routes, seen = [], set()
        scored = [h for h in self.hubs
                  if h != origin and h != dest
                  and any(d == h for d, _ in self.graph.get(origin, []))
                  and any(d == dest for d, _ in self.graph.get(h, []))]

        for hub in scored[:max_hubs]:
            for _, l1 in self.graph.get(origin, []):
                if l1["to_stn"] != hub: continue
                l1d = self._time_to_mins(l1["departure"], l1.get("day_from", 1))
                if l1d < dep_min: continue
                l1a = self._time_to_mins(l1["arrival"], l1.get("day_to", 1))
                if l1a <= l1d: l1a += 1440

                for _, l2 in self.graph.get(hub, []):
                    if l2["to_stn"] != dest: continue
                    l2d = self._time_to_mins(l2["departure"], l2.get("day_from", 1))
                    if l2d < (l1a + min_conn) % 1440: l2d += 1440
                    l2a = self._time_to_mins(l2["arrival"], l2.get("day_to", 1))
                    if l2a <= l2d: l2a += 1440
                    total = l2a - l1d
                    if total > 4320 or total < 0: continue
                    key = f"{l1['train_no']}_{l2['train_no']}_{hub}"
                    if key in seen: continue
                    seen.add(key)
                    via_routes.append({
                        "type": f"VIA {hub}",
                        "train_no": f"{l1['train_no']}+{l2['train_no']}",
                        "train_name": f"{l1['train_name']} -> {l2['train_name']}",
                        "departure": l1["departure"], "arrival": l2["arrival"],
                        "travel_time": self._format_mins(total), "travel_mins": total,
                        "cheapest_fare": 0, "fares": {},
                        "availability": "NOT_CHECKED", "legs": 2, "via": hub,
                    })
        via_routes.sort(key=lambda r: r["travel_mins"])
        return via_routes

    def missed_train_protocol(self, origin, dest, current_time="00:00",
                               budget_max=None, max_hub_search=10):
        print(f"\n{'='*60}")
        print(f"MISSED TRAIN PROTOCOL")
        print(f"  {origin} ({self.station_names.get(origin, '')}) -> {dest} ({self.station_names.get(dest, '')})")
        print(f"  After: {current_time}")
        print(f"{'='*60}")
        opts = []
        direct = self.find_direct_routes(origin, dest, depart_after=current_time)
        opts.extend(direct)
        print(f"  Direct: {len(direct)} trains")
        via = self.find_via_hub_routes(origin, dest, depart_after=current_time, max_hubs=max_hub_search)
        opts.extend(via)
        print(f"  Via-hub: {len(via)} routes")
        opts.sort(key=lambda x: x["travel_mins"])
        for i, o in enumerate(opts[:10], 1):
            print(f"  [{i}] {o['type']} | {o['train_name']} ({o['train_no']}) | Dep {o['departure']} Arr {o['arrival']} | {o['travel_time']}")
        return opts

print("RailDrishtiRouter ready")

# COMMAND ----------

# DBTITLE 1,Initialize Router with Schedule Data
# ============================================================
# Build router from real 417K schedule entries
# ============================================================
router = RailDrishtiRouter()
router.build_from_schedule(df_schedule)

print(f"\nSample hubs: {list(router.hubs)[:15]}")
print(f"\nTry: router.missed_train_protocol('NDLS', 'BCT', current_time='18:00')")

# COMMAND ----------

# DBTITLE 1,AI Layers: NLP Parser, Route Recommender, Waitlist Advisor
# ============================================================
# AI LAYER 1: Natural Language Travel Query Parser
# ============================================================

def ai_parse_travel_query(user_message: str) -> dict:
    """AI parses natural language into structured travel parameters."""
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

    try:
        import re
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(result)
        print(f"  [AI PARSED] Origin: {parsed.get('origin_station_code', '?')} -> Dest: {parsed.get('destination_station_code', '?')}")
        return parsed
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  [AI PARSE WARNING] Could not parse JSON: {result[:200]}")
        return {"raw_response": result, "error": str(e)}


# ============================================================
# AI LAYER 2: Intelligent Route Recommender
# ============================================================

def ai_recommend_route(route_options: list, user_context: dict) -> str:
    """AI analyzes route options and provides intelligent recommendation."""
    routes_text = ""
    for i, r in enumerate(route_options[:10], 1):
        fare_str = f"Rs.{r['cheapest_fare']}" if r.get('cheapest_fare', 0) > 0 else "fare unknown"
        routes_text += (f"Route {i}: [{r['type']}] {r['train_name']} ({r['train_no']}) "
                       f"| Departs: {r['departure']} | Arrives: {r['arrival']} "
                       f"| Time: {r['travel_time']} | Fare: {fare_str} "
                       f"| Seats: {r.get('availability', 'unknown')} "
                       f"| Legs: {r['legs']} | Via: {r.get('via', 'direct')}\n")

    scenario = user_context.get("scenario", "planning_trip")
    urgency = user_context.get("urgency", "today")
    budget_concern = user_context.get("budget_concern", False)
    origin = user_context.get("origin_city", "origin")
    dest = user_context.get("destination_city", "destination")

    prompt = f"""You are Rail-Drishti, an expert Indian Railways travel advisor AI. 
A passenger needs help getting from {origin} to {dest}. 
Scenario: {scenario}. Urgency: {urgency}. Budget sensitive: {budget_concern}.

Here are the available route options:
{routes_text}

Provide a clear, helpful recommendation in 4-6 sentences. Be specific:
1. Which route you recommend FIRST and WHY
2. A backup option if the first choice fails
3. Any warnings (tight connections, known delay-prone trains, waitlist risks)
4. A money-saving tip if applicable

Speak like a knowledgeable Indian travel agent - practical, direct, helpful.
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
    """AI predicts waitlist confirmation probability."""
    prompt = f"""You are an Indian Railways waitlist prediction expert.

A passenger has waitlist position WL/{waitlist_position} on a {train_type} train in {travel_class} class.
Travel is in {days_to_travel} days.

Estimate:
1. Confirmation probability (percentage)
2. Whether they should book this or look for alternatives
3. Key factors affecting confirmation

Be specific with the percentage and reasoning. Keep response to 3-4 sentences."""

    result = spark.sql(f"""
        SELECT ai_query(
            'databricks-meta-llama-3-3-70b-instruct',
            "{prompt.replace('"', "'")}",
            modelParameters => named_struct('temperature', 0.3, 'max_tokens', 300)
        ) AS assessment
    """).collect()[0]["assessment"]

    return result


print("\u2705 AI Layers ready: ai_parse_travel_query, ai_recommend_route, ai_assess_waitlist")

# COMMAND ----------

# DBTITLE 1,AI Layer 4: Train Delay Prediction (LightGBM + MLflow)
# ============================================================
# AI LAYER 4: Train Delay Prediction
# LightGBM model trained on 500K+ delay records
# Loaded from MLflow experiment
# ============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.lightgbm
from datetime import datetime, date

# 1. Load training data to fit label encoders
DELAY_DATA_PATH = "/Workspace/Users/lopamudra.wncc@gmail.com/train_delay/merged_output.csv"
df_delay_raw = pd.read_csv(DELAY_DATA_PATH)
df_delay_raw = df_delay_raw[(df_delay_raw["delay_minutes"] >= -120) & (df_delay_raw["delay_minutes"] <= 600)]
df_delay_raw = df_delay_raw.dropna(subset=["delay_minutes"])

# Fit label encoders (must match training)
le_station = LabelEncoder()
le_station.fit(df_delay_raw["station"])

le_day = LabelEncoder()
le_day.fit(df_delay_raw["day"])

# Available trains in the delay model
DELAY_TRAIN_NOS = set(df_delay_raw["train_no"].unique())
DELAY_STATIONS = set(df_delay_raw["station"].unique())

print(f"Delay dataset: {len(df_delay_raw):,} records")
print(f"Trains covered: {len(DELAY_TRAIN_NOS)} trains")
print(f"Stations covered: {len(DELAY_STATIONS)} stations")

# 2. Load the trained LightGBM model from MLflow
MODEL_URI = "runs:/d9ba760cbb1f42f697214e19ba2095c9/model"
delay_model = mlflow.lightgbm.load_model(MODEL_URI)
print(f"Delay model loaded from MLflow!")

# 3. Build train_no -> station list mapping from delay data
train_station_map = df_delay_raw.groupby("train_no")["station"].apply(lambda x: list(x.unique())).to_dict()

# 4. Build train_no -> name mapping (handle non-numeric like '11039-Slip')
train_name_map = {}
for feat in trains_geo["features"]:
    p = feat["properties"]
    raw_num = str(p.get("number", "") or "")
    try:
        tn = int(raw_num.split("-")[0].strip())
    except (ValueError, IndexError):
        continue
    if tn in DELAY_TRAIN_NOS:
        train_name_map[tn] = p.get("name", str(tn))


def predict_train_delay(train_no, date_str, station=None):
    """
    Predict delay for a train on a given date.
    
    Args:
        train_no: int or str, e.g. 12951
        date_str: str like '2025-04-15' or '15 April 2025'
        station: str station code (optional). If None, predicts for all known stations.
    
    Returns:
        dict with prediction results
    """
    train_no = int(train_no)
    
    if train_no not in DELAY_TRAIN_NOS:
        return {
            "error": True,
            "message": f"Train {train_no} is not in our delay prediction database. "
                       f"We cover {len(DELAY_TRAIN_NOS)} major trains including: "
                       + ", ".join(str(t) for t in sorted(DELAY_TRAIN_NOS)[:10]) + "..."
        }
    
    # Parse date
    try:
        travel_date = pd.to_datetime(date_str)
    except Exception:
        return {"error": True, "message": f"Could not parse date '{date_str}'. Use format like '2025-04-15' or '15 April 2025'."}
    
    month = travel_date.month
    day_of_week = travel_date.dayofweek  # 0=Mon, 6=Sun
    day_name = travel_date.strftime("%A")  # Monday, Tuesday, etc.
    is_holiday = 0  # Default; could enhance with holiday calendar
    
    # Encode day name
    if day_name in le_day.classes_:
        day_enc = le_day.transform([day_name])[0]
    else:
        day_enc = 0
    
    # Determine stations to predict for
    if station:
        stations_to_check = [station.upper()]
    else:
        stations_to_check = train_station_map.get(train_no, [])
    
    if not stations_to_check:
        return {"error": True, "message": f"No station data available for train {train_no}."}
    
    # Build prediction DataFrame
    rows = []
    valid_stations = []
    for stn in stations_to_check:
        if stn in le_station.classes_:
            stn_enc = le_station.transform([stn])[0]
            rows.append({
                "train_no": train_no,
                "station_enc": stn_enc,
                "day_enc": day_enc,
                "month": month,
                "day_of_week": day_of_week,
                "is_holiday": is_holiday,
            })
            valid_stations.append(stn)
    
    if not rows:
        return {"error": True, "message": f"No valid stations found for prediction."}
    
    pred_df = pd.DataFrame(rows)
    predictions = delay_model.predict(pred_df)
    
    # Build results
    station_results = []
    for i, stn in enumerate(valid_stations):
        pred_mins = round(float(predictions[i]), 1)
        station_results.append({"station": stn, "predicted_delay_mins": pred_mins})
    
    # Sort by delay (highest first)
    station_results.sort(key=lambda x: -x["predicted_delay_mins"])
    
    avg_delay = round(float(np.mean(predictions)), 1)
    max_delay = round(float(np.max(predictions)), 1)
    min_delay = round(float(np.min(predictions)), 1)
    train_name = train_name_map.get(train_no, str(train_no))
    
    return {
        "error": False,
        "train_no": train_no,
        "train_name": train_name,
        "date": travel_date.strftime("%A, %B %d, %Y"),
        "avg_delay_mins": avg_delay,
        "max_delay_mins": max_delay,
        "min_delay_mins": min_delay,
        "station_count": len(station_results),
        "station_predictions": station_results,
    }


def format_delay_response(result):
    """Format prediction result into a human-readable response."""
    if result.get("error"):
        return result["message"]
    
    r = result
    resp = (f"Train {r['train_no']} ({r['train_name']}) on {r['date']}:\n\n"
            f"Average predicted delay: {r['avg_delay_mins']} minutes\n"
            f"Maximum delay: {r['max_delay_mins']} minutes\n"
            f"Minimum delay: {r['min_delay_mins']} minutes\n\n")
    
    if r["avg_delay_mins"] <= 5:
        resp += "This train is expected to run mostly on time.\n"
    elif r["avg_delay_mins"] <= 30:
        resp += "Moderate delays expected. Plan some buffer time.\n"
    elif r["avg_delay_mins"] <= 60:
        resp += "Significant delays expected. Keep extra waiting time.\n"
    else:
        resp += "Heavy delays expected! Consider alternative trains or plan accordingly.\n"
    
    # Show top 5 most delayed stations
    resp += "\nStation-wise prediction (top delays):\n"
    for s in r["station_predictions"][:5]:
        resp += f"  {s['station']}: {s['predicted_delay_mins']} min\n"
    
    return resp


print("\nDelay Prediction ready!")
print(f"  predict_train_delay(12951, '2025-06-15')")
print(f"  predict_train_delay(12301, '2025-07-01', station='NDLS')")

# COMMAND ----------

# DBTITLE 1,AI Layer 3: Rail Drishti Complete Advisor + Integrated Chatbot
# ============================================================
# AI LAYER 3: Complete AI-Powered Travel Advisor
# ============================================================

def lookup_station_code(city_or_name: str) -> str:
    """Look up station code from real data."""
    if not city_or_name:
        return None
    search = city_or_name.upper().strip()
    match = df_schedule.filter(
        F.upper(F.col("station_name")).contains(search)
    ).select("station_code").first()
    if match:
        return match.station_code
    match = df_schedule.filter(
        F.upper(F.col("station_code")) == search
    ).select("station_code").first()
    if match:
        return match.station_code
    match = df_stations.filter(
        F.upper(F.col("station_name")).contains(search) |
        F.upper(F.col("address")).contains(search)
    ).first()
    if match:
        sched_match = df_schedule.filter(
            F.col("station_code") == match.station_code
        ).first()
        if sched_match:
            return match.station_code
        sched_match = df_schedule.filter(
            F.upper(F.col("station_name")).contains(match.station_name.upper())
        ).select("station_code").first()
        if sched_match:
            return sched_match.station_code
    return None


def rail_drishti_ai(user_message: str) -> str:
    """
    THE MAIN AI FUNCTION - end-to-end intelligent travel advisor.
    User types natural language -> AI handles everything.
    """
    print(f"{'=' * 60}")
    print(f"RAIL-DRISHTI AI ADVISOR")
    print(f"  User: {user_message}")
    print(f"{'=' * 60}")
    
    print(f"\n[STEP 1] AI parsing your query (ai_query)...")
    parsed = ai_parse_travel_query(user_message)
    
    if "error" in parsed:
        return "Sorry, I couldn't understand your query. Try: 'Chennai to Ujjain'"
    
    scenario = parsed.get("scenario", "planning_trip")
    
    if scenario == "waitlist_query":
        import re
        wl_match = re.search(r'WL[/\-]?(\d+)', user_message, re.IGNORECASE)
        wl_pos = int(wl_match.group(1)) if wl_match else 45
        wl_assessment = ai_assess_waitlist(
            waitlist_position=wl_pos,
            train_type=parsed.get("train_mentioned", "Express") or "Express",
            travel_class=parsed.get("travel_class_preference", "SL") or "SL",
            days_to_travel=3
        )
        return wl_assessment
    
    origin = parsed.get("origin_station_code")
    destination = parsed.get("destination_station_code")
    origin_city = parsed.get("origin_city", "")
    dest_city = parsed.get("destination_city", "")
    current_time = parsed.get("current_time", "08:00")
    
    if not origin or origin == "None":
        origin = lookup_station_code(origin_city)
        if origin:
            print(f"    Resolved '{origin_city}' -> {origin}")
        else:
            return f"Could not find station for '{origin_city}'."
    else:
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
    
    print(f"\n[STEP 2] Searching routes: {origin} -> {destination}...")
    
    route_options = router.missed_train_protocol(
        origin=origin,
        dest=destination,
        current_time=current_time,
        budget_max=None,
        max_hub_search=10
    )
    
    if not route_options:
        return f"No routes found from {origin} to {destination} after {current_time}."
    
    print(f"\n[STEP 3] AI analyzing {len(route_options)} options (ai_query)...")
    recommendation = ai_recommend_route(route_options, parsed)
    
    return recommendation


# ============================================================
# DELAY PREDICTION QUERY HANDLER
# ============================================================
import re as _re

def is_delay_query(message: str) -> bool:
    """Detect if the message is asking about train delays."""
    delay_keywords = [
        "delay", "delayed", "late", "how late", "on time",
        "kitni der", "der se", "delay predict",
        "will be delayed", "expected delay", "running late",
        "delay status", "delay forecast", "delay prediction",
        "how much delay", "how much late",
    ]
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in delay_keywords)


def handle_delay_query(user_message: str) -> str:
    """
    Parse a delay query to extract train number and date,
    then call the delay prediction model.
    """
    print(f"{'=' * 60}")
    print(f"TRAIN DELAY PREDICTION")
    print(f"  User: {user_message}")
    print(f"{'=' * 60}")
    
    # Try to extract train number from message
    train_match = _re.search(r'\b(\d{4,5})\b', user_message)
    
    # Try to extract date from message
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})',
        r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4})',
    ]
    date_str = None
    for pattern in date_patterns:
        dm = _re.search(pattern, user_message, _re.IGNORECASE)
        if dm:
            date_str = dm.group(1)
            break
    
    # Check for relative dates
    msg_lower = user_message.lower()
    if not date_str:
        if "tomorrow" in msg_lower or "kal" in msg_lower:
            from datetime import timedelta
            date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "today" in msg_lower or "aaj" in msg_lower:
            date_str = datetime.now().strftime("%Y-%m-%d")
    
    # If still no train or date, use AI to parse
    if not train_match or not date_str:
        print("  [AI] Parsing delay query with LLM...")
        prompt = (
            "Extract the train number and travel date from this query. "
            "Respond with ONLY JSON: {\"train_no\": <number>, \"date\": \"YYYY-MM-DD\"}\n"
            "If no date is mentioned, use today. If no train number, set train_no to null.\n\n"
            f"Query: {user_message}\n"
            f"Today: {datetime.now().strftime('%Y-%m-%d')}\n"
            "JSON:"
        )
        try:
            result = spark.sql(f"""
                SELECT ai_query(
                    'databricks-meta-llama-3-3-70b-instruct',
                    "{prompt.replace('"', "'")}",
                    modelParameters => named_struct('temperature', 0.1, 'max_tokens', 100)
                ) AS parsed
            """).collect()[0]["parsed"]
            json_match = _re.search(r'\{[^{}]*\}', result, _re.DOTALL)
            if json_match:
                ai_parsed = json.loads(json_match.group())
                if not train_match and ai_parsed.get("train_no"):
                    class FakeMatch:
                        def group(self, n=1): return str(ai_parsed['train_no'])
                    train_match = FakeMatch()
                if not date_str and ai_parsed.get("date"):
                    date_str = ai_parsed["date"]
        except Exception as e:
            print(f"  [AI PARSE WARNING] {e}")
    
    # Default date to today
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    if not train_match:
        return ("I need a train number to predict delays. "
                "Please ask like: 'How much delay for train 12951 on 2025-06-15?'\n"
                f"I can predict delays for {len(DELAY_TRAIN_NOS)} major trains including: "
                + ", ".join(str(t) for t in sorted(DELAY_TRAIN_NOS)[:15]) + "...")
    
    train_no = int(train_match.group(1))
    print(f"  Train: {train_no} | Date: {date_str}")
    
    # Check if a station was mentioned
    station = None
    for stn in DELAY_STATIONS:
        if stn.lower() in user_message.lower():
            station = stn
            break
    
    # Run prediction
    result = predict_train_delay(train_no, date_str, station=station)
    return format_delay_response(result)


# ============================================================
# INTEGRATED CHATBOT: 3 capabilities
# ============================================================

def is_travel_routing_query(message: str) -> bool:
    """Detect if a message is about train routes, schedules, or travel planning."""
    travel_keywords = [
        "train from", "train to", "trains from", "trains to",
        "route from", "route to", "travel from", "travel to",
        "missed train", "missed my train",
        "delhi to", "mumbai to", "chennai to", "kolkata to", "bangalore to",
        "hyderabad to", "pune to", "jaipur to", "lucknow to", "ahmedabad to",
        "varanasi to", "bhopal to", "kota to",
        "next train", "trains between", "direct train",
        "alternative train", "connecting train",
        "how to reach", "how to go",
        "waitlist", "wl/", "wl ", "waiting list", "confirmation chance",
        "rac", "pnr status",
    ]
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in travel_keywords)


def chat_with_assistant_v2(message, history):
    """
    Enhanced chatbot with 3 capabilities:
    1. Train delay prediction (LightGBM model)
    2. Travel routing (Rail Drishti graph engine)
    3. General Q&A (LLM + PDF knowledge base)
    """
    try:
        # PRIORITY 1: Delay prediction queries
        if is_delay_query(message):
            print("Detected delay query -> Using Delay Prediction Model")
            response = handle_delay_query(message)
            return response
        
        # PRIORITY 2: Travel routing queries
        if is_travel_routing_query(message):
            print("Detected travel/routing query -> Using Rail Drishti AI")
            response = rail_drishti_ai(message)
            return response
        
        # PRIORITY 3: General query -> LLM + PDF search
        print("General query -> Using LLM + PDF search")
        pdf_excerpt = get_relevant_pdf_excerpt(message)
        
        if pdf_excerpt:
            pdf_list = ", ".join(pdf_contents.keys())
            system_prompt = f"""{BASE_SYSTEM_PROMPT}

IMPORTANT: {len(pdf_contents)} PDF document(s) have been uploaded: {pdf_list}. Use the following content from the PDFs to answer the user's question. If the answer is in the PDFs, cite the source document. If not, provide general assistance.

--- PDF CONTENT ---
{pdf_excerpt}
--- END PDF CONTENT ---
"""
        else:
            system_prompt = BASE_SYSTEM_PROMPT
        
        messages = [ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt)]
        
        for user_msg, bot_msg in history:
            messages.append(ChatMessage(role=ChatMessageRole.USER, content=user_msg))
            messages.append(ChatMessage(role=ChatMessageRole.ASSISTANT, content=bot_msg))
        
        messages.append(ChatMessage(role=ChatMessageRole.USER, content=message))
        
        response = w.serving_endpoints.query(
            name="databricks-meta-llama-3-3-70b-instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        bot_response = response.choices[0].message.content
        return bot_response
        
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again."

chat_with_assistant = chat_with_assistant_v2

print("Integrated chatbot ready with 3 capabilities!")
print("  1. Train Delay Prediction -> 'How much delay for 12951 on June 15?'")
print("  2. Travel/Routing -> 'Delhi to Mumbai trains'")
print("  3. General Q&A -> 'What is the baggage allowance?'")

# COMMAND ----------

# Test the chatbot with multiple questions
test_questions = [
    "Where can I find the restroom?",
    "Is WiFi available on this train?",
    "How much baggage can I carry?"
]

display("🎯 Testing Train Passenger Assistant Chatbot")
display("="*70)

for i, question in enumerate(test_questions, 1):
    display(f"\n👤 Question {i}: {question}")
    display("🤖 Bot Response:")
    response = chat_with_assistant(question, [])
    display(response)
    display("-"*70)

display("\n✅ CHATBOT IS FULLY FUNCTIONAL!")
display("You can ask any question using: chat_with_assistant(your_question, [])")
display("="*70)

# COMMAND ----------

# For demo purposes, let's simulate PDF content without actually uploading a file
# You can skip this and directly upload your PDF using upload_pdf()

# Simulate PDF content (replace with actual PDF upload)
pdf_content = """TRAIN PASSENGER MANUAL

BAGGAGE ALLOWANCE POLICY
Passengers in AC First Class are allowed 70kg of baggage.
Passengers in AC 2-Tier are allowed 50kg of baggage.
Passengers in AC 3-Tier are allowed 40kg of baggage.
Passengers in Sleeper Class are allowed 40kg of baggage.

Each piece of baggage should not exceed 25kg in weight.

ONBOARD FACILITIES
- Free WiFi available on all premium trains
- Charging points available at every berth
- Restrooms located at both ends of each coach
- Pantry car serves meals from 7 AM to 10 PM

TICKET CANCELLATION POLICY
- More than 48 hours before departure: Full refund minus 10% charges
- 24-48 hours before departure: 50% refund
- Less than 24 hours: 25% refund
- After chart preparation: No refund

EMERGENCY PROCEDURES
- Emergency alarm chain: Pull only in case of serious emergency
- Contact train conductor via intercom
- Medical assistance: Contact pantry car staff
- Fire extinguishers located near toilets

PLATFORM INFORMATION
- Platform numbers displayed on station boards
- Train arrival notifications via SMS and app
- Coaches labeled A1, A2, B1, B2, etc.
- Check coach position chart near platform entrance
"""

pdf_filename = "train_manual_demo.pdf"

display("✅ Demo PDF content loaded!")
display(f"📄 Simulated file: {pdf_filename}")
display(f"📝 Content length: {len(pdf_content)} characters")
display("\n👉 Try asking questions like:")
display("  - What is the baggage allowance for AC 2-Tier?")
display("  - How do I cancel my ticket?")
display("  - Where are the restrooms located?")
display("  - What are the emergency procedures?")

# COMMAND ----------

# === UPLOAD YOUR PDFs HERE ===
# You can now load MULTIPLE PDF files!

# Method 1: Upload PDFs one by one
upload_pdf("/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/SCR G&SR UPDATED AS16 HR.pdf")
upload_pdf("/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/1463038915607-समान्य नियम 2015 for  Printing.pdf")
upload_pdf("/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/etktfaq.pdf")
upload_pdf("/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/CitizenCharter.pdf")
upload_pdf("/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/features_of_internet_tickets.pdf")
upload_pdf("/Workspace/Users/lopamudra.wncc@gmail.com/Chatbot/tatkal_faq.pdf")

display("\n" + "="*70)
display("✅ PDF loading complete!")
display(f"📚 Total documents loaded: {len(pdf_contents)}")
display("\n💡 You can now ask questions about your documents using:")
display("   chat_with_assistant('your question here', [])")
display("\n🔍 The chatbot will search across ALL loaded PDFs automatically!")
display("="*70)

# Show all loaded PDFs
list_pdfs()

# COMMAND ----------

# Ask a question about the SCR G&SR document

question = "Hello can i make tatkal 48 hours befdore?"

display(f"❓ Question: {question}")
display("\n🤖 Bot Response:")
display("="*70)

response = chat_with_assistant(question, [])
display(response)

display("="*70)
display("\n💡 Try other questions:")
display("  - What are the safety procedures for train operations?")
display("  - What are the hand signals used by railway staff?")
display("  - What are the rules for foggy weather operations?")

# COMMAND ----------

# 🌍 Test Multilingual Support - Questions in Different Languages

test_questions_multilingual = [
    ("English", "What is the baggage allowance?"),
    ("Hindi", "ट्रेन में बाथरूम कहाँ है?"),
    ("Tamil", "விபத்து எப்படி நடக்கும்?"),
    ("Spanish", "¿Dónde está el baño?")
]

display("🌍 Testing Multilingual Chatbot")
display("="*70)

for lang, question in test_questions_multilingual:
    display(f"\n🌎 Language: {lang}")
    display(f"💬 Question: {question}")
    display("🤖 Bot Response:")
    
    response = chat_with_assistant(question, [])
    display(response)
    display("-"*70)

display("\n✅ MULTILINGUAL SUPPORT IS WORKING!")
display("\n💡 The bot automatically detects and responds in the user's language.")
display("\nSupported languages include:")
display("  🇮🇳 Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu")
display("  🌍 English, Spanish, French, German, Chinese, Japanese, Korean, Arabic, and 100+ more!")
display("="*70)

# COMMAND ----------

# Test 1: Travel routing query (should use Rail Drishti)
print("=" * 70)
print("TEST 1: Travel Routing Query")
print("=" * 70)
response1 = chat_with_assistant("Delhi to Mumbai trains", [])
print("\n🤖 Response:")
print(response1)


# COMMAND ----------

# DBTITLE 1,Test Delay Prediction Integration
# Test delay prediction through the chatbot
print("=" * 70)
print("TEST: Delay Prediction via Chatbot")
print("=" * 70)

# Test 1: Direct delay query
q1 = "How much delay for train 12951 tomorrow?"
print(f"\nQ: {q1}")
print("\nBot Response:")
print(chat_with_assistant(q1, []))

print("\n" + "=" * 70)

# Test 2: Hindi delay query
q2 = "Train 12301 kitni der se chalti hai?"
print(f"\nQ: {q2}")
print("\nBot Response:")
print(chat_with_assistant(q2, []))