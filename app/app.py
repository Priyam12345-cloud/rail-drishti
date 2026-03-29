import json, os, re, sys, traceback, html
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import uvicorn

print("[BOOT] Starting Rail Drishti...", flush=True)

SDK_OK = False
w = None
try:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    w = WorkspaceClient()
    SDK_OK = True
    print("[BOOT] SDK OK", flush=True)
except Exception as e:
    print(f"[BOOT] SDK: {e}", flush=True)

LLM = "databricks-meta-llama-3-3-70b-instruct"
schedules, stations, trains = [], [], []

def sdk_load(path):
    resp = w.workspace.download(path)
    return json.load(resp)

def load_file(name):
    fp = f"/Workspace/Users/lopamudra.wncc@gmail.com/Data/{name}"
    if os.path.exists(fp):
        with open(fp) as f: return json.load(f)
    elif SDK_OK:
        print(f"[DATA] Downloading {name} via SDK", flush=True)
        return sdk_load(f"/Users/lopamudra.wncc@gmail.com/Data/{name}")
    raise FileNotFoundError(name)

station_coords = {}
try:
    print("[DATA] Loading...", flush=True)
    schedules = load_file("schedules.json")
    print(f"[DATA] Schedules: {len(schedules)}", flush=True)
    sg = load_file("stations.json")
    for ft in sg["features"]:
        p = ft["properties"]; g = ft.get("geometry")
        c = g["coordinates"] if g and g.get("coordinates") else [None, None]
        stations.append({"code": p.get("code",""), "name": p.get("name",""), "state": p.get("state",""), "zone": p.get("zone","")})
        if c[0] is not None and c[1] is not None:
            station_coords[p.get("code","")] = (c[1], c[0])
    print(f"[DATA] Stations: {len(stations)}, Coords: {len(station_coords)}", flush=True)
    tg = load_file("trains.json")
    for ft in tg["features"]:
        p = ft["properties"]
        trains.append({"number": str(p.get("number","")), "name": p.get("name",""),
            "from_code": p.get("from_station_code",""), "from_name": p.get("from_station_name",""),
            "to_code": p.get("to_station_code",""), "to_name": p.get("to_station_name",""),
            "zone": p.get("zone",""), "type": p.get("type",""),
            "third_ac": int(p.get("third_ac",0) or 0), "sleeper": int(p.get("sleeper",0) or 0),
            "second_ac": int(p.get("second_ac",0) or 0), "first_ac": int(p.get("first_ac",0) or 0),
            "first_class": int(p.get("first_class",0) or 0), "chair_car": int(p.get("chair_car",0) or 0),
            "departure": p.get("departure",""), "arrival": p.get("arrival",""),
            "duration_h": p.get("duration_h",0), "duration_m": p.get("duration_m",0)})
    print(f"[DATA] Trains: {len(trains)}", flush=True)
except Exception as e:
    print(f"[DATA] Error: {e}", flush=True)

# ── DELAY STATS (pre-computed 28KB JSON) ──
delay_by_train = {}
delay_by_day = {}
try:
    ds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "delay_stats.json")
    if os.path.exists(ds_path):
        print("[DATA] Loading delay stats...", flush=True)
        with open(ds_path) as f:
            delay_by_train = json.load(f)
    elif SDK_OK:
        print("[DATA] Downloading delay stats via SDK...", flush=True)
        delay_by_train = sdk_load("/Users/lopamudra.wncc@gmail.com/UI/rail-drishti-app/delay_stats.json")
    print(f"[DATA] Delay: {len(delay_by_train)} trains", flush=True)
    for tno, d in delay_by_train.items():
        for day, avg in d.get("days", {}).items():
            if day not in delay_by_day: delay_by_day[day] = []
            delay_by_day[day].append(avg)
    for day in delay_by_day:
        delay_by_day[day] = round(sum(delay_by_day[day])/len(delay_by_day[day]), 1) if delay_by_day[day] else 0
except Exception as e:
    print(f"[DATA] Delay error: {e}", flush=True)

# ── STATS ──
total_trains = len(trains)
total_stations = len(stations)
total_schedules = len(schedules)
total_delay_trains = len(delay_by_train)
zone_data = Counter(t["zone"] for t in trains).most_common(10)
type_data = Counter(t["type"] for t in trains if t["type"]).most_common(8)
cls_stats = {}
for cls in ["sleeper","third_ac","second_ac","first_ac","chair_car","first_class"]:
    cls_stats[cls] = sum(1 for t in trains if t.get(cls,0)>0)
stn_ts = defaultdict(set)
stn_nm = {}
for s in schedules:
    c = s.get("station_code", "")
    stn_ts[c].add(s.get("train_number", ""))
    if c and s.get("station_name"): stn_nm[c] = s["station_name"]
top_stns = sorted([(c, stn_nm.get(c,c), len(t)) for c,t in stn_ts.items()], key=lambda x: -x[2])[:15]
top_delayed = sorted([(tno, d["avg"], d["cnt"]) for tno,d in delay_by_train.items() if d["cnt"]>50], key=lambda x: -x[1])[:10]
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
day_delays = [(d, delay_by_day.get(d, 0) if not isinstance(delay_by_day.get(d, 0), list) else 0) for d in day_order]
train_name_map = {t["number"]: t["name"] for t in trains}
train_type_map = {t["number"]: t["type"] for t in trains}
print("[STATS] Done", flush=True)

# ── ROUTER (INDEX-BASED + CITY CLUSTERS) ──
class Router:
    CITY_CLUSTERS = {
        "BCT":["BCT","CSTM","LTT","BDTS","DR","DDR"],"CSTM":["BCT","CSTM","LTT","BDTS","DR","DDR"],
        "LTT":["BCT","CSTM","LTT","BDTS","DR","DDR"],"BDTS":["BCT","CSTM","LTT","BDTS","DR","DDR"],
        "DR":["BCT","CSTM","LTT","BDTS","DR","DDR"],"DDR":["BCT","CSTM","LTT","BDTS","DR","DDR"],
        "NDLS":["NDLS","DEE","NZM","ANVT","DLI"],"DEE":["NDLS","DEE","NZM","ANVT","DLI"],
        "NZM":["NDLS","DEE","NZM","ANVT","DLI"],"ANVT":["NDLS","DEE","NZM","ANVT","DLI"],
        "DLI":["NDLS","DEE","NZM","ANVT","DLI"],
        "HWH":["HWH","SDAH","KOAA"],"SDAH":["HWH","SDAH","KOAA"],"KOAA":["HWH","SDAH","KOAA"],
        "MAS":["MAS","MS","MSB","TBM"],
        "SBC":["SBC","KSR","YPR","BNCE"],"KSR":["SBC","KSR","YPR","BNCE"],"YPR":["SBC","KSR","YPR","BNCE"],
        "SC":["SC","HYB","KCG"],"HYB":["SC","HYB","KCG"],
        "PNBE":["PNBE","PNC","RJPB"],"PNC":["PNBE","PNC","RJPB"],"RJPB":["PNBE","PNC","RJPB"],
        "LKO":["LKO","LJN"],"LJN":["LKO","LJN"],"ADI":["ADI","SBIJ"],"JP":["JP","JPS"],
    }
    def __init__(self):
        self.graph = defaultdict(list); self.hubs = set(); self.names = {}
        self.train_stops = {}; self.stn2trains = defaultdict(set)
    def _xc(self, code): return self.CITY_CLUSTERS.get(code, [code])
    def build(self, scheds):
        if not scheds: return
        st = defaultdict(set)
        for s in scheds: st[s.get("station_code","")].add(s.get("train_number",""))
        self.hubs = {c for c,t in st.items() if len(t)>=10}
        for s in scheds:
            c, n = s.get("station_code",""), s.get("station_name","")
            if c and n: self.names[c] = n
        trs = defaultdict(list)
        for s in scheds: trs[s.get("train_number","")].append(s)
        for tno, stops in trs.items():
            stops.sort(key=lambda x: (x.get("day",0) or 0, x.get("id",0) or 0))
            tn = stops[0].get("train_name","") or tno
            if len(stops) < 2: continue
            sl = []
            for s in stops:
                sc = s.get("station_code","")
                if sc:
                    sl.append({"code":sc,"dep":s.get("departure","") or "","arr":s.get("arrival","") or "","day":s.get("day",1) or 1,"tn":tn})
                    self.stn2trains[sc].add(tno)
            self.train_stops[tno] = sl
            for i in range(len(sl)-1):
                s1,s2 = sl[i],sl[i+1]
                if s1["code"]!=s2["code"]:
                    self.graph[s1["code"]].append((s2["code"],{"train_no":tno,"train_name":tn,"from_stn":s1["code"],"to_stn":s2["code"],"departure":s1["dep"],"arrival":s2["arr"],"day_from":s1["day"],"day_to":s2["day"]}))
        print(f"[ROUTER] {len(self.graph)} stns, {len(self.hubs)} hubs, {len(self.train_stops)} trains indexed", flush=True)
    def _t2m(self, t, day=1):
        try:
            p = str(t).strip().replace(".",":").split(":")
            return (day-1)*1440+int(p[0])*60+(int(p[1]) if len(p)>1 else 0)
        except: return (day-1)*1440
    def _fmt(self, m):
        if m<=0: m+=1440
        return f"{m//60}h {m%60}m"
    def search(self, ori, dst, after="00:00", date=None):
        td = datetime.strptime(date,"%Y-%m-%d") if date else datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
        dm = self._t2m(after); routes=[]; seen=set()
        # INDEX-BASED direct search with city clusters
        oc = set(self._xc(ori)); dc = set(self._xc(dst))
        to = set(); td2 = set()
        for c in oc: to |= self.stn2trains.get(c, set())
        for c in dc: td2 |= self.stn2trains.get(c, set())
        for tno in to & td2:
            sl = self.train_stops[tno]
            oi,os2 = None,None
            for i,s in enumerate(sl):
                if s["code"] in oc: oi=i; os2=s; break
            if oi is None: continue
            ds = None
            for j in range(oi+1, len(sl)):
                if sl[j]["code"] in dc: ds=sl[j]; break
            if ds is None: continue
            dep=self._t2m(os2["dep"],os2["day"]); arr=self._t2m(ds["arr"],ds["day"])
            if dep<dm: continue
            tr=arr-dep
            if tr<=0: tr+=1440
            k=tno
            if k in seen: continue
            seen.add(k)
            ad = ds["code"]
            tl = "DIRECT" if ad==dst else f"DIRECT ({ad})"
            routes.append({"type":tl,"train_no":k,"train_name":os2.get("tn",tno),"departure":os2["dep"],"arrival":ds["arr"],"travel_time":self._fmt(tr),"travel_mins":tr,"depart_date":(td+timedelta(minutes=dep)).strftime("%b %d %H:%M"),"arrive_date":(td+timedelta(minutes=dep+tr)).strftime("%b %d %H:%M")})
        # INDEX-BASED via-hub with cap
        hc = set()
        for tno in to:
            sl = self.train_stops[tno]; oi2=None
            for i,s in enumerate(sl):
                if s["code"] in oc: oi2=i; break
            if oi2 is None: continue
            for j in range(oi2+1,len(sl)):
                sc=sl[j]["code"]
                if sc in self.hubs and sc not in oc and sc not in dc: hc.add(sc)
        sh = [(h,len(self.stn2trains.get(h,set())&td2)) for h in hc if self.stn2trains.get(h,set())&td2]
        sh.sort(key=lambda x:-x[1]); via_count=0
        for hub,_ in sh[:8]:
            l1t = to & self.stn2trains.get(hub,set())
            for tno1 in l1t:
                sl1=self.train_stops[tno1]; oi3,hi=None,None
                for i,s in enumerate(sl1):
                    if oi3 is None and s["code"] in oc: oi3=i
                    elif oi3 is not None and s["code"]==hub: hi=i; break
                if oi3 is None or hi is None: continue
                so,sh2=sl1[oi3],sl1[hi]
                l1d=self._t2m(so["dep"],so["day"])
                if l1d<dm: continue
                l1a=self._t2m(sh2["arr"],sh2["day"])
                if l1a<=l1d: l1a+=1440
                l2t = self.stn2trains.get(hub,set()) & td2
                for tno2 in l2t:
                    sl2=self.train_stops[tno2]; hj,dj=None,None
                    for j,s in enumerate(sl2):
                        if hj is None and s["code"]==hub: hj=j
                        elif hj is not None and s["code"] in dc: dj=j; break
                    if hj is None or dj is None: continue
                    shd,sd=sl2[hj],sl2[dj]
                    l2do=self._t2m(shd["dep"],shd["day"]); l2ao=self._t2m(sd["arr"],sd["day"])
                    if l2ao<=l2do: l2ao+=1440
                    ld=l2ao-l2do; l2d=self._t2m(shd["dep"])
                    while l2d<l1a+45: l2d+=1440
                    if l2d-l1a>2880: continue
                    l2a=l2d+ld; total=l2a-l1d
                    if total>4320 or total<0: continue
                    k=tno1+"_"+tno2+"_"+hub
                    if k in seen: continue
                    seen.add(k)
                    routes.append({"type":"VIA "+hub,"train_no":tno1+"+"+tno2,"train_name":so.get("tn",tno1)+" > "+shd.get("tn",tno2),"departure":so["dep"],"arrival":sd["arr"],"travel_time":self._fmt(total),"travel_mins":total,"depart_date":(td+timedelta(minutes=l1d)).strftime("%b %d %H:%M"),"arrive_date":(td+timedelta(minutes=l2a)).strftime("%b %d %H:%M")})
                    via_count+=1
                    if via_count>=200: break
                if via_count>=200: break
            if via_count>=200: break
        routes.sort(key=lambda x: x["travel_mins"])
        return routes

print("[ROUTER] Building...", flush=True)
router = Router()
router.build(schedules)
total_hubs = len(router.hubs)

# ── FARE ENGINE ──
import math
PER_KM = {"GEN": 0.22, "SL": 0.45, "CC": 1.10, "3AC": 1.20, "2AC": 1.85, "1AC": 3.10}
RSV_CHG = {"GEN": 0, "SL": 40, "CC": 40, "3AC": 40, "2AC": 50, "1AC": 60}
SF_CHG = {"GEN": 0, "SL": 30, "CC": 30, "3AC": 45, "2AC": 50, "1AC": 75}
TYPE_MULT = {"Rajdhani":1.35, "Shatabdi":1.30, "Duronto":1.25, "Vande Bharat":1.40, "Garib Rath":0.75, "Tejas":1.45, "Humsafar":1.15}
ALL_CLASSES = ["GEN", "SL", "CC", "3AC", "2AC", "1AC"]
def haversine_km(la1, lo1, la2, lo2):
    if any(v is None for v in (la1,lo1,la2,lo2)): return None
    R=6371; dl=math.radians(la2-la1); dn=math.radians(lo2-lo1)
    a=math.sin(dl/2)**2+math.cos(math.radians(la1))*math.cos(math.radians(la2))*math.sin(dn/2)**2
    return R*2*math.asin(math.sqrt(a))
def route_distance_km(tno, fc, tc):
    sl = router.train_stops.get(tno, [])
    if not sl:
        c1,c2=station_coords.get(fc),station_coords.get(tc)
        if c1 and c2:
            d=haversine_km(c1[0],c1[1],c2[0],c2[1])
            return round(d*1.3) if d else None
        return None
    fi,ti=None,None
    for i,s in enumerate(sl):
        if s["code"]==fc and fi is None: fi=i
        if s["code"]==tc and fi is not None: ti=i; break
    if fi is None or ti is None or fi>=ti:
        c1,c2=station_coords.get(fc),station_coords.get(tc)
        if c1 and c2:
            d=haversine_km(c1[0],c1[1],c2[0],c2[1])
            return round(d*1.3) if d else None
        return None
    tot=0.0
    for i in range(fi,ti):
        c1=station_coords.get(sl[i]["code"]); c2=station_coords.get(sl[i+1]["code"])
        if c1 and c2:
            seg=haversine_km(c1[0],c1[1],c2[0],c2[1])
            if seg: tot+=seg
    return round(tot*1.2) if tot>0 else None
def calc_fare(dk, cls, tt="", tno=""):
    if not dk or dk<=0: return None
    base=round(dk*PER_KM.get(cls,0.45))
    for kw,ml in TYPE_MULT.items():
        if kw.lower() in (tt or "").lower(): base=round(base*ml); break
    rsv=RSV_CHG.get(cls,40)
    sf=SF_CHG.get(cls,30) if tno.startswith("12") or tno.startswith("22") else 0
    return base+rsv+sf
def enrich_routes_fares(rts):
    for r in rts:
        tnos=r["train_no"].split("+"); is_via="VIA" in r.get("type","")
        # Determine actual origin/dest from train stops
        ori_c=set(Router.CITY_CLUSTERS.get(r.get("_ori",""),[r.get("_ori","")]))
        dst_c=set(Router.CITY_CLUSTERS.get(r.get("_dst",""),[r.get("_dst","")]))
        if is_via and len(tnos)>=2:
            # Find actual stations from train stops
            hub=r.get("type","").replace("VIA ","")
            sl1=router.train_stops.get(tnos[0],[]); sl2=router.train_stops.get(tnos[1],[])
            ao,ah1,ah2,ad="","","",""
            for s in sl1:
                if s["code"] in ori_c and not ao: ao=s["code"]
                if ao and s["code"]==hub: ah1=s["code"]; break
            for s in sl2:
                if s["code"]==hub and not ah2: ah2=s["code"]
                if ah2 and s["code"] in dst_c: ad=s["code"]; break
            d1=route_distance_km(tnos[0],ao,hub) if ao and hub else None
            d2=route_distance_km(tnos[1],hub,ad) if hub and ad else None
            t1,t2=train_type_map.get(tnos[0],""),train_type_map.get(tnos[1],"")
            fares={}
            for cls in ALL_CLASSES:
                f1=calc_fare(d1,cls,t1,tnos[0]) if d1 else 0
                f2=calc_fare(d2,cls,t2,tnos[1]) if d2 else 0
                if f1 or f2: fares[cls]=(f1 or 0)+(f2 or 0)
            r["fares"]=fares; r["dist"]=(d1 or 0)+(d2 or 0)
        else:
            sl=router.train_stops.get(tnos[0],[]); ao,ad="",""
            for s in sl:
                if s["code"] in ori_c and not ao: ao=s["code"]
                if ao and s["code"] in dst_c: ad=s["code"]; break
            dist=route_distance_km(tnos[0],ao,ad) if ao and ad else None
            tt=train_type_map.get(tnos[0],"")
            fares={}
            for cls in ALL_CLASSES:
                f=calc_fare(dist,cls,tt,tnos[0])
                if f: fares[cls]=f
            r["fares"]=fares; r["dist"]=dist or 0
    return rts
print("[FARE] Engine ready", flush=True)

# ── HELPERS ──
def llm_query(prompt, temp=0.5, mx=500):
    if not SDK_OK: return "AI needs serving endpoint. Add in App Settings."
    try:
        r = w.serving_endpoints.query(name=LLM, messages=[ChatMessage(role=ChatMessageRole.USER,content=prompt)], temperature=temp, max_tokens=mx)
        return r.choices[0].message.content
    except Exception as e: return f"Error: {e}"

CITY_ALIASES = {
    "MUMBAI":"BCT","DELHI":"NDLS","NEW DELHI":"NDLS","KOLKATA":"HWH","CALCUTTA":"HWH",
    "CHENNAI":"MAS","MADRAS":"MAS","BANGALORE":"SBC","BENGALURU":"SBC","HYDERABAD":"SC",
    "PATNA":"PNBE","AHMEDABAD":"ADI","JAIPUR":"JP","LUCKNOW":"LKO","VARANASI":"BSB",
    "PUNE":"PUNE","BHOPAL":"BPL","KOTA":"KOTA","NAGPUR":"NGP","KANPUR":"CNB",
    "GUWAHATI":"GHY","JAMMU":"JAT","AMRITSAR":"ASR","CHANDIGARH":"CDG","DEHRADUN":"DDN",
    "INDORE":"INDB","COIMBATORE":"CBE","MADURAI":"MDU","TRIVANDRUM":"TVC",
    "COCHIN":"ERS","KOCHI":"ERS","GOA":"MAO","VIZAG":"VSKP","VISAKHAPATNAM":"VSKP",
    "RANCHI":"RNC","RAIPUR":"R","BHUBANESWAR":"BBS","PURI":"PURI","AGRA":"AGC",
    "ALLAHABAD":"ALD","PRAYAGRAJ":"PRYJ","UJJAIN":"UJN","GWALIOR":"GWL",
}

def lookup(city):
    if not city: return None
    s = city.upper().strip()
    if s in CITY_ALIASES: return CITY_ALIASES[s]
    for sc in schedules:
        if s == (sc.get("station_code","") or "").upper(): return sc.get("station_code")
        if s in (sc.get("station_name","") or "").upper(): return sc.get("station_code")
    return None

def predict_delay(train_no):
    tno = str(train_no).strip()
    if tno not in delay_by_train: return None
    d = delay_by_train[tno]
    return {"avg": d["avg"], "max": d["max"], "min": d["min"], "median": d["med"], "count": d["cnt"],
            "top_stations": d.get("top_stns", []), "by_day": sorted(d.get("days", {}).items(), key=lambda x: -x[1])}


def svg_bar(data, w=400, h=200, c1="#667eea", c2="#764ba2"):
    if not data: return ""
    mx = max(v for _,v in data) or 1
    bw = min(40, (w-40) // len(data) - 4)
    bars = ""
    for i,(lbl,val) in enumerate(data):
        bh = int(val/mx * (h-50))
        x = 40 + i * (bw + 4)
        y = h - 30 - bh
        cl = c1 if i%2==0 else c2
        bars += f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="4" fill="{cl}" opacity="0.85"/>'
        bars += f'<text x="{x+bw//2}" y="{y-4}" text-anchor="middle" font-size="10" fill="#555">{val}</text>'
        bars += f'<text x="{x+bw//2}" y="{h-16}" text-anchor="middle" font-size="9" fill="#888">{lbl[:3]}</text>'
    return f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-height:{h}px">{bars}</svg>'

def dcol(m):
    if m<=5: return "#43e97b"
    if m<=15: return "#38f9d7"
    if m<=30: return "#ffd200"
    if m<=60: return "#f7971e"
    return "#f5576c"

def dlbl(m):
    if m<=5: return "On Time"
    if m<=15: return "Slight"
    if m<=30: return "Moderate"
    if m<=60: return "Significant"
    return "Heavy"

chat_history = []
ratings_data = []
chat_lang = 'English'

CSS = '''
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;background:#f5f6fa;min-height:100vh}
.ctr{max-width:1300px;margin:0 auto;padding:20px}
.hdr{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);padding:30px 40px;border-radius:16px;color:white;margin-bottom:20px}
.hdr h1{font-size:2.4em;font-weight:800;color:#e9c46a}
.hdr p{font-size:1.05em;opacity:0.9;margin-top:8px;color:#a8dadc}
.nav{display:flex;justify-content:center;gap:6px;margin-bottom:24px;flex-wrap:wrap}
.nav a{padding:10px 18px;border-radius:10px;text-decoration:none;font-weight:600;font-size:0.95em;transition:all 0.2s}
.nav a:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(102,126,234,0.3)}
.card{background:white;border-radius:16px;padding:24px;box-shadow:0 2px 16px rgba(0,0,0,0.06);margin-bottom:20px}
.card h3{color:#1a1a2e;margin-bottom:14px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
.st{border-radius:16px;padding:24px;text-align:center;color:white;transition:transform 0.2s}
.st:hover{transform:translateY(-3px)}
.st h3{font-size:2.4em;font-weight:700;margin:0;color:inherit}
.st p{font-size:0.9em;opacity:0.9;margin-top:4px}
table{width:100%;border-collapse:collapse}
th{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:12px 16px;text-align:left;font-weight:600}
td{padding:10px 16px;border-bottom:1px solid #f0f0f0}
tr:hover{background:#f8f9ff}
input[type=text],textarea{width:100%;padding:12px 16px;border:2px solid #e0e0e0;border-radius:10px;font-size:1em;margin:4px 0;transition:border-color 0.2s}
input:focus,textarea:focus{border-color:#667eea;outline:none;box-shadow:0 0 0 3px rgba(102,126,234,0.1)}
.btn{background:linear-gradient(135deg,#667eea,#764ba2);color:white;border:none;padding:12px 28px;border-radius:10px;font-size:1em;cursor:pointer;font-weight:600;transition:all 0.2s}
.btn:hover{opacity:0.9;transform:translateY(-1px)}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.bu{display:flex;justify-content:flex-end;margin:8px 0}
.bu div{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:12px 18px;border-radius:18px 18px 4px 18px;max-width:75%;line-height:1.5}
.bb{display:flex;justify-content:flex-start;margin:8px 0}
.bb div{background:white;border:1px solid #e0e0e0;padding:12px 18px;border-radius:18px 18px 18px 4px;max-width:75%;line-height:1.5}
.tag{display:inline-block;padding:3px 12px;border-radius:12px;font-size:0.85em;font-weight:600}
.tag-d{background:#43e97b;color:white}.tag-v{background:#f5576c;color:white}.tag-b{background:#667eea;color:white}
.fare-card{background:linear-gradient(135deg,#f8f9ff,#eef1ff);border:2px solid #667eea;border-radius:14px;padding:16px;margin:6px 0;text-align:center}
.fare-card h4{color:#667eea;font-size:1.05em;margin:0}
.fare-card .price{font-size:1.8em;font-weight:700;color:#1a1a2e;margin:4px 0}
.dbar{display:flex;align-items:center;gap:8px;padding:5px 0}
.dbar .lbl{min-width:80px;font-weight:500;color:#555}
.dbar .bar{height:24px;border-radius:6px;display:flex;align-items:center;padding-left:8px;color:white;font-size:0.85em;font-weight:600;min-width:30px}
.ib{background:linear-gradient(135deg,#e8f4f8,#d4e6f1);border-left:4px solid #2980b9;padding:16px 20px;border-radius:0 12px 12px 0;margin:10px 0}
.rc{background:linear-gradient(135deg,#ffecd2,#fcb69f);padding:20px;border-radius:12px;margin:8px 0}
@media(max-width:768px){.stats{grid-template-columns:repeat(2,1fr)}.g2,.g3{grid-template-columns:1fr}}
'''

def page(title, content, active="home"):
    tabs = [("home","Home"),("chat","AI Assistant"),("routes","Routes"),("delay","Delays"),("ratings","Ratings"),("search","Search")]
    nav = ""
    for tid, tlbl in tabs:
        if tid == active:
            sty = "background:linear-gradient(135deg,#667eea,#764ba2);color:white"
        else:
            sty = "background:#f0f0f0;color:#333"
        nav += '<a href="/' + tid + '" style="' + sty + '">' + tlbl + '</a>'
    hs = f"{total_trains:,} Trains | {total_stations:,} Stations | {total_schedules:,} Schedules | {total_delay_trains} Delay-Tracked"
    return ('<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        + '<title>Rail Drishti - ' + html.escape(title) + '</title><style>' + CSS + '</style></head><body>'
        + '<div class="ctr"><div class="hdr"><h1>Rail-Drishti</h1><p>AI Railways Intelligence | ' + hs + '</p></div>'
        + '<div class="nav">' + nav + '</div>' + content
        + '<div style="text-align:center;padding:20px;color:#888;font-size:0.9em">Rail-Drishti | Databricks AI + Datameet + Indian Railways Open Data</div>'
        + '</div></body></html>')

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok", "trains": total_trains, "stations": total_stations}

@app.get("/", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
async def home_page():
    s = '<div class="stats">'
    s += f'<div class="st" style="background:linear-gradient(135deg,#667eea,#764ba2)"><h3>{total_trains:,}</h3><p>Trains</p></div>'
    s += f'<div class="st" style="background:linear-gradient(135deg,#f093fb,#f5576c)"><h3>{total_stations:,}</h3><p>Stations</p></div>'
    s += f'<div class="st" style="background:linear-gradient(135deg,#43e97b,#38f9d7);color:#1a1a2e"><h3>{total_hubs:,}</h3><p>Hub Junctions</p></div>'
    s += f'<div class="st" style="background:linear-gradient(135deg,#f7971e,#ffd200);color:#1a1a2e"><h3>{total_delay_trains}</h3><p>Delay-Tracked</p></div></div>'
    tbl = '<div class="card"><h3>Top Connected Stations</h3><table><tr><th>#</th><th>Station</th><th>Code</th><th>Trains</th></tr>'
    for i,(c,n,cnt) in enumerate(top_stns,1):
        tbl += f'<tr><td>{i}</td><td>{html.escape(n)}</td><td><code>{html.escape(c)}</code></td><td style="color:#667eea;font-weight:600">{cnt}</td></tr>'
    tbl += '</table></div>'
    bars = '<div class="card"><h3>Zone Distribution</h3>' + svg_bar([(z[:5],cnt) for z,cnt in zone_data]) + '</div>'
    types = '<div class="card"><h3>Train Types</h3>'
    for tp, cnt in type_data:
        pct = int(cnt/total_trains*100) if total_trains else 0
        types += f'<div class="dbar"><span class="lbl">{html.escape(tp or "Other")}</span><div class="bar" style="width:{max(pct,5)}%;background:linear-gradient(90deg,#667eea,#764ba2)">{cnt}</div></div>'
    types += '</div>'
    cls = '<div class="card"><h3>Class Availability</h3>'
    for k,lbl in [("sleeper","SL"),("third_ac","3A"),("second_ac","2A"),("first_ac","1A"),("chair_car","CC")]:
        cnt = cls_stats.get(k,0); pct = int(cnt/total_trains*100) if total_trains else 0
        cls += f'<div class="dbar"><span class="lbl">{lbl}</span><div class="bar" style="width:{max(pct,3)}%;background:linear-gradient(90deg,#43e97b,#38f9d7);color:#1a1a2e">{cnt}</div></div>'
    cls += '</div>'
    dly = '<div class="card"><h3>Average Delay by Day</h3>'
    dly += svg_bar([(d[:3],v) for d,v in day_delays], c1="#f7971e", c2="#f5576c") if day_delays else '<p style="color:#888">No data</p>'
    dly += '</div>'
    return page("Home", s + '<div class="g2">' + tbl + bars + '</div><div class="g3">' + types + cls + dly + '</div>', "home")

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    msgs = ""
    for u, b in chat_history:
        msgs += f'<div class="bu"><div>{html.escape(u)}</div></div><div class="bb"><div>{b}</div></div>'
    if not msgs:
        msgs = '<div style="text-align:center;color:#888;padding:60px">Ask me anything about Indian Railways!<br><small>Routes, delays, tickets, tatkal, baggage... in any language!</small></div>'
    # Language selector
    langs = [("English","EN"),("Hindi","HI"),("Tamil","TA"),("Telugu","TE"),("Bengali","BN"),("Marathi","MR"),("Gujarati","GU"),("Kannada","KN"),("Malayalam","ML"),("Punjabi","PA"),("Urdu","UR"),("Spanish","ES"),("French","FR")]
    lang_bar = '<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-wrap:wrap"><span style="font-weight:600;color:#555;font-size:0.9em">Language:</span>'
    for lname, lcode in langs:
        if lname == chat_lang:
            sty = "background:linear-gradient(135deg,#667eea,#764ba2);color:white;border:none"
        else:
            sty = "background:white;color:#555;border:1px solid #ddd"
        lang_bar += f'<form method="POST" action="/chat_lang" style="display:inline"><input type="hidden" name="lang" value="{lname}"><button type="submit" style="{sty};padding:5px 12px;border-radius:16px;cursor:pointer;font-size:0.82em;font-weight:500;transition:all 0.2s">{lname}</button></form>'
    lang_bar += '</div>'
    c = '<div class="card">'
    c += lang_bar
    c += f'<div style="min-height:400px;max-height:500px;overflow-y:auto;padding:16px;background:#fafafa;border-radius:12px;margin-bottom:16px">{msgs}</div>'
    c += '<form method="POST" action="/chat" style="display:flex;gap:8px"><input type="text" name="message" placeholder="Ask about trains..." style="flex:1" autofocus autocomplete="off"><button type="submit" class="btn">Send</button></form>'
    c += '<div style="display:flex;gap:6px;margin-top:10px;flex-wrap:wrap">'
    for ex in ["Delhi to Mumbai trains","Delay for train 12951","What is tatkal?","Baggage allowance?"]:
        c += '<form method="POST" action="/chat" style="display:inline"><input type="hidden" name="message" value="' + html.escape(ex) + '"><button type="submit" style="background:#f0f0f0;color:#333;border:1px solid #ddd;border-radius:20px;padding:6px 14px;cursor:pointer;font-size:0.85em">' + html.escape(ex) + '</button></form>'
    c += '</div><form method="POST" action="/chat_clear" style="margin-top:8px"><button type="submit" style="background:none;border:1px solid #ddd;padding:6px 16px;border-radius:8px;cursor:pointer;color:#888">Clear</button></form></div>'
    return page("AI Assistant", c, "chat")

@app.post("/chat", response_class=HTMLResponse)
async def chat_submit(message: str = Form("")):
    global chat_history
    if message.strip():
        reply = "Add serving endpoint in App Settings."
        delay_kw = ["delay","delayed","late","on time","how late","running late"]
        route_kw = ["train from","train to","delhi to","mumbai to","chennai to","how to reach","trains between","missed train","kolkata to","bangalore to"]
        ml = message.lower()
        if SDK_OK:
            try:
                if any(k in ml for k in delay_kw):
                    tm = re.search(r"\b(\d{4,5})\b", message)
                    if tm:
                        tno = tm.group(1); pred = predict_delay(tno)
                        if pred:
                            tn = train_name_map.get(tno, tno)
                            reply = f"<b>Train {tno} ({html.escape(tn)})</b><br>Avg delay: <b>{pred['avg']} min</b> | Median: {pred['median']} min<br>Range: {pred['min']} to {pred['max']} min ({pred['count']:,} obs)<br>"
                            if pred.get("top_stations"):
                                reply += "<br><b>Most delayed stations:</b><br>"
                                for s,v in pred["top_stations"]:
                                    reply += f"&nbsp;&nbsp;{html.escape(s)}: {v} min<br>"
                        else:
                            lang_note = f" Respond in {chat_lang}." if chat_lang != "English" else ""
                            reply = llm_query(f"User asks about delay for train {tm.group(1)}. Not in our 98-train database. Give general advice.{lang_note}", 0.5, 300)
                    else:
                        reply = llm_query(f"User asks: {message}. Help with delay info.", 0.5, 300)
                elif any(k in ml for k in route_kw):
                    prompt = 'Parse Indian Railways query to JSON: origin_station_code, destination_station_code, origin_city, destination_city, current_time. Delhi:NDLS Mumbai:BCT Kolkata:HWH Chennai:MAS. Query: "' + message + '"'
                    r = llm_query(prompt, 0.1, 400)
                    try:
                        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', r, re.DOTALL)
                        p = json.loads(m.group()) if m else {}
                    except: p = {}
                    ori = p.get("origin_station_code") or lookup(p.get("origin_city",""))
                    dst = p.get("destination_station_code") or lookup(p.get("destination_city",""))
                    if ori and dst:
                        rts = router.search(ori, dst, p.get("current_time","08:00"), datetime.now().strftime("%Y-%m-%d"))
                        if rts:
                            lines = [f"Route {i}: [{r['type']}] {r['train_name']} | {r['depart_date']}>{r['arrive_date']} | {r['travel_time']}" for i,r in enumerate(rts[:8],1)]
                            on = router.names.get(ori, ori); dn = router.names.get(dst, dst)
                            lang_note2 = f" Respond in {chat_lang}." if chat_lang != "English" else ""
                            rec = llm_query("Recommend best route in 4 sentences:" + lang_note2 + "\n" + "\n".join(lines), 0.5, 500)
                            reply = f"<b>{html.escape(on)} to {html.escape(dn)}</b> ({len(rts)} routes)<br><br>{rec}"
                        else: reply = f"No routes {ori} to {dst}."
                    else: reply = "Could not find stations. Try: Delhi to Mumbai trains"
                else:
                    lang_instr = f" IMPORTANT: Always respond in {chat_lang} language." if chat_lang != "English" else ""
                    msgs = [ChatMessage(role=ChatMessageRole.SYSTEM, content="You are Rail-Drishti, a helpful Indian Railways assistant. Be concise." + lang_instr)]
                    for u, b in chat_history[-5:]:
                        msgs.append(ChatMessage(role=ChatMessageRole.USER, content=u))
                        msgs.append(ChatMessage(role=ChatMessageRole.ASSISTANT, content=b))
                    msgs.append(ChatMessage(role=ChatMessageRole.USER, content=message))
                    r = w.serving_endpoints.query(name=LLM, messages=msgs, temperature=0.7, max_tokens=500)
                    reply = r.choices[0].message.content
            except Exception as e: reply = f"Error: {e}"
        chat_history.append((message, reply))
    return await chat_page()

@app.post("/chat_clear", response_class=HTMLResponse)
async def chat_clear():
    global chat_history; chat_history = []; return await chat_page()

@app.post("/chat_lang", response_class=HTMLResponse)
async def chat_set_lang(lang: str = Form("English")):
    global chat_lang
    chat_lang = lang
    return await chat_page()

@app.get("/routes", response_class=HTMLResponse)
async def routes_page():
    today = datetime.now().strftime("%Y-%m-%d")
    c = '<div class="card"><h3>Find Train Routes</h3><form method="POST" action="/routes"><div style="display:grid;grid-template-columns:2fr 2fr 1fr 1fr;gap:12px">'
    c += '<div><label>Origin</label><input type="text" name="origin" placeholder="Delhi / NDLS"></div>'
    c += '<div><label>Destination</label><input type="text" name="destination" placeholder="Mumbai / BCT"></div>'
    c += '<div><label>After</label><input type="text" name="time" value="08:00"></div>'
    c += f'<div><label>Date</label><input type="text" name="date" value="{today}"></div>'
    c += '</div><br><button type="submit" class="btn">Find Routes</button></form></div>'
    return page("Routes", c, "routes")

@app.post("/routes", response_class=HTMLResponse)
async def routes_search(origin: str = Form(""), destination: str = Form(""), time: str = Form("08:00"), date: str = Form("")):
    if not origin or not destination: return page("Routes", '<div class="card">Enter both origin and destination.</div>', "routes")
    oc = lookup(origin) or origin.upper().strip(); dc = lookup(destination) or destination.upper().strip()
    on = router.names.get(oc, origin); dn = router.names.get(dc, destination)
    rts = router.search(oc, dc, time, date or datetime.now().strftime("%Y-%m-%d"))
    for r in rts: r["_ori"]=oc; r["_dst"]=dc
    rts = enrich_routes_fares(rts)
    hdr = f'<div class="card" style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white"><h3 style="color:#e9c46a">{html.escape(on)} to {html.escape(dn)}</h3><p style="opacity:0.8">{len(rts)} routes</p></div>'
    if not rts: return page("Routes", hdr + '<div class="card">No routes found.</div>', "routes")
    direct = [r for r in rts if "DIRECT" in r.get("type","")]
    connecting = [r for r in rts if "VIA" in r.get("type","")]
    hdr += f'<p style="color:#e9c46a">{len(direct)} Direct \u2022 {len(connecting)} Connecting</p>'
    cards = ""
    for i, r in enumerate(rts[:20], 1):
        is_d = "DIRECT" in r.get("type","")
        tag_cls = "tag-d" if is_d else "tag-v"
        tag_lbl = "NON-STOP" if is_d else html.escape(r["type"])
        fares = r.get("fares", {})
        dist = r.get("dist", 0)
        ttype = train_type_map.get(r["train_no"].split("+")[0], "")
        pills = ""
        for cls in ["GEN","SL","CC","3AC","2AC","1AC"]:
            v = fares.get(cls, 0)
            if v > 0:
                bg = {"GEN":"#555","SL":"#2196f3","CC":"#4caf50","3AC":"#ff9800","2AC":"#e91e63","1AC":"#9c27b0"}.get(cls,"#888")
                pills += f'<span style="background:{bg};color:white;padding:3px 8px;border-radius:4px;margin:2px;font-size:12px;display:inline-block"><b>{cls}</b> \u20b9{v:,}</span> '
        fare_html = f'<div style="margin-top:8px">{pills}</div>' if pills else ""
        meta = f'<span style="color:#888;font-size:13px">{dist:,} km</span>'
        if ttype:
            meta += f' <span style="background:#1a1a2e;color:#e9c46a;padding:2px 6px;border-radius:3px;font-size:11px">{html.escape(ttype)}</span>'
        bdr = "#4caf50" if is_d else "#ff9800"
        cards += f'<div class="card" style="margin:8px 0;padding:14px;border-left:4px solid {bdr}">'
        cards += f'<div style="display:flex;justify-content:space-between;align-items:center">'
        cards += f'<div><span class="tag {tag_cls}">{tag_lbl}</span> <b>{html.escape(r["train_name"][:50])}</b> <small style="color:#888">({html.escape(r["train_no"])})</small></div>'
        cards += f'<div><span class="tag tag-b">{r["travel_time"]}</span></div></div>'
        cards += f'<div style="display:flex;justify-content:space-between;margin-top:6px">'
        cards += f'<div><b>{html.escape(r["depart_date"])}</b> \u2192 <b>{html.escape(r["arrive_date"])}</b></div>'
        cards += f'<div>{meta}</div></div>'
        cards += fare_html + '</div>'
    return page("Routes", hdr + cards, "routes")

@app.get("/delay", response_class=HTMLResponse)
async def delay_page():
    c = '<div class="card"><h3>Train Delay Prediction</h3><div class="ib">Powered by 500K+ historical records covering 98 major trains across 382 stations.</div>'
    c += '<form method="POST" action="/delay" style="display:flex;gap:8px;margin-top:16px"><input type="text" name="train_no" placeholder="Train number (e.g. 12951)" style="flex:1"><button type="submit" class="btn">Predict</button></form></div>'
    if top_delayed:
        c += '<div class="card"><h3>Most Delayed Trains</h3><table><tr><th>#</th><th>Train</th><th>Avg Delay</th><th>Status</th></tr>'
        for i,(tno,avg,cnt) in enumerate(top_delayed,1):
            tn = train_name_map.get(tno, tno)
            c += f'<tr><td>{i}</td><td><b>{html.escape(tn)}</b><br><small>{tno}</small></td><td style="font-weight:600;color:{dcol(avg)}">{avg} min</td><td><span class="tag" style="background:{dcol(avg)};color:white">{dlbl(avg)}</span></td></tr>'
        c += '</table></div>'
    if day_delays:
        c += '<div class="card"><h3>Delay by Day of Week</h3>' + svg_bar([(d[:3],v) for d,v in day_delays], c1="#f7971e", c2="#f5576c") + '</div>'
    return page("Delays", c, "delay")

@app.post("/delay", response_class=HTMLResponse)
async def delay_predict(train_no: str = Form("")):
    if not train_no.strip(): return await delay_page()
    pred = predict_delay(train_no.strip())
    if not pred:
        c = f'<div class="card" style="text-align:center;padding:40px"><h3>Train {html.escape(train_no)} Not Found</h3><p style="color:#888">Not in our {total_delay_trains}-train database. Try: ' + ", ".join(list(delay_by_train.keys())[:6]) + '</p></div>'
        return page("Delays", c, "delay")
    tno = train_no.strip(); tn = train_name_map.get(tno, tno); avg = pred["avg"]
    c = f'<div class="card" style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:white"><h3 style="color:#e9c46a">{html.escape(tn)}</h3><p>Train {tno} | {pred["count"]:,} observations</p></div>'
    c += '<div class="stats">'
    c += f'<div class="st" style="background:{dcol(avg)}"><h3>{avg}</h3><p>Avg Delay (min)</p></div>'
    c += f'<div class="st" style="background:linear-gradient(135deg,#667eea,#764ba2)"><h3>{pred["median"]}</h3><p>Median</p></div>'
    c += f'<div class="st" style="background:linear-gradient(135deg,#f093fb,#f5576c)"><h3>{pred["max"]}</h3><p>Max</p></div>'
    c += f'<div class="st" style="background:linear-gradient(135deg,#43e97b,#38f9d7);color:#1a1a2e"><h3>{pred["min"]}</h3><p>Min</p></div></div>'
    if pred.get("by_day"):
        c += '<div class="card"><h3>Delay by Day</h3>' + svg_bar([(d[:3],v) for d,v in pred["by_day"]], c1="#667eea", c2="#764ba2") + '</div>'
    if pred.get("top_stations"):
        c += '<div class="card"><h3>Most Delayed Stations</h3>'
        for s,v in pred["top_stations"]:
            sn = stn_nm.get(s, s); pct = min(int(v/max(pred["max"],1)*100), 100)
            c += f'<div class="dbar"><span class="lbl">{html.escape(sn)}</span><div class="bar" style="width:{max(pct,8)}%;background:{dcol(v)}">{v}m</div></div>'
        c += '</div>'
    if avg<=5: rec="Excellent punctuality. Very reliable!"
    elif avg<=15: rec="Generally punctual with minor delays."
    elif avg<=30: rec="Moderate delays. Add 30 min buffer."
    elif avg<=60: rec="Frequently delayed. Plan extra time."
    else: rec="Chronically delayed. Consider alternatives."
    c += f'<div class="ib"><b>Recommendation:</b> {rec}</div>'
    return page("Delays", c, "delay")



@app.get("/ratings", response_class=HTMLResponse)
async def ratings_page():
    reviews = ""
    if ratings_data:
        sm = {}
        for r in ratings_data:
            k = r["tn"]+" - "+r["nm"]
            if k not in sm: sm[k]={"rs":[],"cs":[]}
            sm[k]["rs"].append(r["r"]); sm[k]["cs"].append(r)
        for t,d in sorted(sm.items(), key=lambda x:-sum(x[1]["rs"])/len(x[1]["rs"])):
            avg=round(sum(d["rs"])/len(d["rs"]),1)
            reviews += f'<div class="rc"><b>{html.escape(t)}</b> {avg}/5 ({len(d["rs"])})'
            for c2 in d["cs"][-3:]:
                reviews += f'<div style="background:white;border-radius:8px;padding:10px;margin-top:6px;border-left:3px solid #f7971e"><b>{html.escape(str(c2["rv"]))}</b> {c2["r"]}/5<p style="margin:2px 0;color:#555">{html.escape(str(c2["c"]))}</p></div>'
            reviews += '</div>'
    else: reviews = '<div style="text-align:center;color:#888;padding:40px">No ratings yet.</div>'
    c = '<div class="g2"><div class="card"><h3>Rate a Train</h3><form method="POST" action="/ratings">'
    c += '<label>Train Number</label><input type="text" name="tno" placeholder="12952">'
    c += '<label>Train Name</label><input type="text" name="tnm" placeholder="Rajdhani">'
    c += '<label>Rating (1-5)</label><input type="text" name="rating" value="4">'
    c += '<label>Review</label><textarea name="comment" rows="3" placeholder="Your experience..."></textarea>'
    c += '<label>Name</label><input type="text" name="reviewer" value="Anonymous"><br>'
    c += '<button type="submit" class="btn">Submit</button></form></div>'
    c += '<div class="card"><h3>Reviews</h3>' + reviews + '</div></div>'
    return page("Ratings", c, "ratings")

@app.post("/ratings", response_class=HTMLResponse)
async def ratings_submit(tno: str = Form(""), tnm: str = Form(""), rating: str = Form("4"), comment: str = Form(""), reviewer: str = Form("Anonymous")):
    if tno:
        if not tnm: tnm = train_name_map.get(tno, "Train "+tno)
        try: rv = max(1,min(5,int(rating)))
        except: rv = 4
        ratings_data.append({"tn":tno,"nm":tnm,"r":rv,"c":comment or "No comment","rv":reviewer or "Anonymous"})
    return await ratings_page()

@app.get("/search", response_class=HTMLResponse)
async def search_page():
    c = '<div class="card"><h3>Search Trains</h3><form method="POST" action="/search" style="display:flex;gap:8px"><input type="text" name="query" placeholder="Rajdhani, 12952..." style="flex:1" autofocus><button type="submit" class="btn">Search</button></form></div>'
    return page("Search", c, "search")

@app.post("/search", response_class=HTMLResponse)
async def search_submit(query: str = Form("")):
    res = [t for t in trains if query.strip() in t["number"] or query.upper().strip() in t["name"].upper()][:20] if query else []
    cards = ""
    for t in res:
        dur = str(t["duration_h"])+"h "+str(t["duration_m"])+"m" if t["duration_h"] else "N/A"
        has_d = t["number"] in delay_by_train
        di = ""
        if has_d:
            d = delay_by_train[t["number"]]; di = f' | <span style="color:{dcol(d["avg"])}">Avg delay: {d["avg"]}m</span>'
        fb = []
        for k,l in [("sleeper","SL"),("third_ac","3A"),("second_ac","2A"),("first_ac","1A")]:
            v = t.get(k,0)
            if v: fb.append(f"{l}:Rs.{v}")
        fs = " | ".join(fb)
        cards += f'<div class="card" style="padding:16px"><b style="color:#667eea;font-size:1.1em">{html.escape(t["name"])}</b> <code>{t["number"]}</code>'
        if has_d: cards += ' <span class="tag tag-b">Delay Data</span>'
        cards += f'<br><span style="color:#555">{html.escape(t["from_name"])} to {html.escape(t["to_name"])}</span><br><small>{html.escape(t["zone"])} | {dur}{di}</small>'
        if fs: cards += f'<br><small style="color:#43e97b">{fs}</small>'
        cards += '</div>'
    if not cards and query: cards = f'<div class="card" style="text-align:center;padding:30px">No trains for "{html.escape(query)}".</div>'
    form = '<div class="card"><h3>Search Trains</h3><form method="POST" action="/search" style="display:flex;gap:8px"><input type="text" name="query" value="' + html.escape(query or "") + '" style="flex:1" autofocus><button type="submit" class="btn">Search</button></form></div>'
    if res: form += f'<p style="color:#888;margin-bottom:12px">{len(res)} results</p>'
    return page("Search", form + cards, "search")

print("[LAUNCH] Starting...", flush=True)
port = int(os.environ.get("DATABRICKS_APP_PORT", os.environ.get("UVICORN_PORT", "8080")))
print(f"[LAUNCH] Port: {port}", flush=True)
uvicorn.run(app, host="0.0.0.0", port=port)
