# Databricks notebook source
# MAGIC %md
# MAGIC # Multi-Modal Graph Route Optimizer
# MAGIC ### Minimum Cost Navigation with Mode Constraints
# MAGIC
# MAGIC **Problem:** Find cheapest path from A to B across 5 transport modes:
# MAGIC - **BUS** — Mumbai BEST city bus stops (from `edges.csv`)
# MAGIC - **CENTRAL** — Central Line suburban train
# MAGIC - **WESTERN** — Western Line suburban train
# MAGIC - **HARBOUR** — Harbour Line suburban train
# MAGIC - **TRANS_HARBOUR** — Trans-Harbour Line
# MAGIC
# MAGIC **Constraint:** Max 3 different modes per journey
# MAGIC
# MAGIC **Data Sources:**
# MAGIC - `edges.csv` — Bus stop edges with fare weights
# MAGIC - `routes.json` — Bus route stop sequences
# MAGIC - `train_to_bus_mapping.json` — Train station ↔ bus stop transfers
# MAGIC
# MAGIC **Algorithm:** Modified Dijkstra on multi-layer graph with state = `(cost, node, modes_used)`

# COMMAND ----------

# DBTITLE 1,Build Multi-Modal Transport Graph
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import heapq
from difflib import SequenceMatcher

DATA_DIR = "/Workspace/Users/lopamudra.wncc@gmail.com/Multimodel"

# ============================================================
# 1. LOAD BUS EDGES (edges.csv) - actual weighted connections
# ============================================================
df_edges = pd.read_csv(f"{DATA_DIR}/edges.csv")
df_edges.columns = df_edges.columns.str.strip()
df_edges['start'] = df_edges['start'].str.strip().str.upper()
df_edges['stop'] = df_edges['stop'].str.strip().str.upper()
print(f"Bus edges loaded: {len(df_edges):,}")
print(f"Columns: {list(df_edges.columns)}")
print(f"Fare range: {df_edges['fare'].min()} - {df_edges['fare'].max()}")
print(f"Sample:")
display(df_edges.head(3))

# ============================================================
# 2. LOAD TRAIN-TO-BUS MAPPING (transfer connections)
# ============================================================
with open(f"{DATA_DIR}/train_to_bus_mapping.json", "r") as f:
    mapping = json.load(f)

print(f"\nTrain lines: {list(mapping['routes'].keys())}")
for line_key, line_data in mapping['routes'].items():
    n_stations = len(line_data['stations'])
    n_with_bus = sum(1 for s in line_data['stations'] if s['nearest_bus_stops'])
    print(f"  {line_data['name']}: {n_stations} stations, {n_with_bus} with bus transfers")

# ============================================================
# 3. BUILD MULTI-MODAL GRAPH from real data
#    graph[node] = [(neighbor, cost, mode), ...]
# ============================================================
graph = defaultdict(list)
all_nodes = set()
node_modes = defaultdict(set)

# --- 3a. BUS EDGES from edges.csv ---
bus_edge_count = 0
for _, row in df_edges.iterrows():
    a, b = row['start'], row['stop']
    cost = max(int(row['fare']), 1)
    # Bidirectional
    graph[a].append((b, cost, 'BUS'))
    graph[b].append((a, cost, 'BUS'))
    all_nodes.update([a, b])
    node_modes[a].add('BUS')
    node_modes[b].add('BUS')
    bus_edge_count += 1

print(f"\nBus edges: {bus_edge_count:,}")
print(f"Bus stops: {len([n for n in all_nodes if 'BUS' in node_modes[n]])}")

# --- 3b. TRAIN LINE EDGES (consecutive stations from mapping) ---
# Map line keys to mode names
LINE_MODES = {
    'central': 'CENTRAL',
    'western': 'WESTERN',
    'harbour': 'HARBOUR',
    'trans_harbour': 'TRANS_HARBOUR',
}

def train_fare_from_km(km_diff):
    """Approximate Mumbai local train fare from distance."""
    if km_diff <= 5: return 5
    if km_diff <= 10: return 10
    if km_diff <= 20: return 15
    return 20

train_edge_count = 0
for line_key, mode_name in LINE_MODES.items():
    if line_key not in mapping['routes']:
        continue
    stations = mapping['routes'][line_key]['stations']
    for i in range(len(stations) - 1):
        st_a = stations[i]['train_station'].strip().upper()
        st_b = stations[i + 1]['train_station'].strip().upper()
        km_a = stations[i].get('km', 0) or 0
        km_b = stations[i + 1].get('km', 0) or 0
        cost = train_fare_from_km(abs(km_b - km_a))
        
        graph[st_a].append((st_b, cost, mode_name))
        graph[st_b].append((st_a, cost, mode_name))
        all_nodes.update([st_a, st_b])
        node_modes[st_a].add(mode_name)
        node_modes[st_b].add(mode_name)
        train_edge_count += 1

print(f"Train edges: {train_edge_count}")

# --- 3c. TRANSFER EDGES from train_to_bus_mapping ---
TRANSFER_COST = 2  # Walking cost to switch mode
transfer_count = 0

for line_key, mode_name in LINE_MODES.items():
    if line_key not in mapping['routes']:
        continue
    for station in mapping['routes'][line_key]['stations']:
        train_name = station['train_station'].strip().upper()
        bus_stops = station.get('nearest_bus_stops', [])
        for bus_stop in bus_stops:
            bs = bus_stop.strip().upper()
            if bs and bs in all_nodes:  # Only if bus stop exists in our graph
                graph[train_name].append((bs, TRANSFER_COST, 'TRANSFER'))
                graph[bs].append((train_name, TRANSFER_COST, 'TRANSFER'))
                transfer_count += 1

print(f"Transfer edges: {transfer_count} (from mapping file)")

# --- Summary ---
print(f"\n\u2705 Multi-modal graph built from your data files!")
print(f"   Total nodes: {len(all_nodes):,}")
print(f"   Total edges: {bus_edge_count + train_edge_count + transfer_count:,}")
print(f"   Modes: BUS, CENTRAL, WESTERN, HARBOUR, TRANS_HARBOUR")
print(f"   Multi-modal nodes: {sum(1 for n in all_nodes if len(node_modes[n]) > 1)}")

# Show transfer points
multi = [(n, node_modes[n]) for n in all_nodes if len(node_modes[n]) > 1]
if multi:
    print(f"\nTransfer hubs (2+ modes):")
    for name, modes in sorted(multi, key=lambda x: -len(x[1]))[:15]:
        print(f"   {name}: {', '.join(sorted(modes))}")

def fuzzy_score(a, b):
    return SequenceMatcher(None, a.upper(), b.upper()).ratio()

# COMMAND ----------

# DBTITLE 1,Constrained Dijkstra - Min Cost with Mode Limit
# ============================================================
# MODIFIED DIJKSTRA: Min cost path with max K modes
# State: (cumulative_cost, current_node, frozenset(modes_used))
# Constraint: len(modes_used) <= max_modes
# ============================================================

def dijkstra_multi_modal(graph, source, target, max_modes=3):
    """
    Find minimum cost path from source to target using at most max_modes
    different transport modes.
    Returns: (total_cost, path, mode_sequence) or None if no path exists
    """
    pq = [(0, source, frozenset(), [source], [])]
    visited = {}
    
    while pq:
        cost, node, modes_used, path, mode_path = heapq.heappop(pq)
        
        if node == target:
            return cost, path, mode_path
        
        state = (node, modes_used)
        if state in visited and visited[state] <= cost:
            continue
        visited[state] = cost
        
        for neighbor, edge_cost, mode in graph[node]:
            if mode == 'TRANSFER':
                new_modes = modes_used
                actual_mode = mode_path[-1] if mode_path else 'WALK'
            else:
                new_modes = modes_used | {mode}
                actual_mode = mode
            
            if len(new_modes) > max_modes:
                continue
            
            new_cost = cost + edge_cost
            new_state = (neighbor, new_modes)
            
            if new_state not in visited or visited[new_state] > new_cost:
                heapq.heappush(pq, (
                    new_cost, neighbor, new_modes,
                    path + [neighbor],
                    mode_path + [actual_mode]
                ))
    
    return None


def find_top_k_routes(graph, source, target, max_modes=3, k=3):
    """Find top-K diverse routes (different mode combinations)."""
    results = []
    seen_mode_combos = set()
    pq = [(0, source, frozenset(), [source], [])]
    visited_counts = defaultdict(int)
    
    while pq and len(results) < k * 5:
        cost, node, modes_used, path, mode_path = heapq.heappop(pq)
        
        if node == target:
            combo = frozenset(modes_used)
            if combo not in seen_mode_combos:
                seen_mode_combos.add(combo)
                results.append((cost, path, mode_path))
                if len(results) >= k:
                    break
            continue
        
        state = (node, modes_used)
        visited_counts[state] += 1
        if visited_counts[state] > 2:
            continue
        
        for neighbor, edge_cost, mode in graph[node]:
            if mode == 'TRANSFER':
                new_modes = modes_used
                actual_mode = mode_path[-1] if mode_path else 'WALK'
            else:
                new_modes = modes_used | {mode}
                actual_mode = mode
            
            if len(new_modes) > max_modes:
                continue
            
            heapq.heappush(pq, (
                cost + edge_cost, neighbor, new_modes,
                path + [neighbor], mode_path + [actual_mode]
            ))
    
    return sorted(results, key=lambda x: x[0])[:k]

print("\u2705 Constrained Dijkstra ready!")
print("   State = (cost, node, frozenset(modes_used))")
print("   Constraint: len(modes_used) <= max_modes")

# COMMAND ----------

# DBTITLE 1,AI Route Navigator - Smart Suggestions
# ============================================================
# AI ROUTE NAVIGATOR
# - Fuzzy stop name matching (handles typos)
# - Generates step-by-step navigation instructions
# - Ranks routes by cost + modes + transfers
# ============================================================

def fuzzy_find_stop(query, all_nodes, threshold=0.6):
    """Find best matching stop name using fuzzy matching."""
    q = query.strip().upper()
    if q in all_nodes:
        return q
    matches = [n for n in all_nodes if q in n or n in q]
    if matches:
        return min(matches, key=len)
    best_score, best_match = 0, None
    for node in all_nodes:
        score = fuzzy_score(q, node)
        if score > best_score:
            best_score, best_match = score, node
    return best_match if best_score >= threshold else None


def format_navigation(cost, path, mode_path):
    """Generate step-by-step navigation instructions."""
    icons = {'BUS': '\U0001f68d', 'CENTRAL': '\U0001f682', 'HARBOUR': '\u2693', 'WALK': '\U0001f6b6'}
    names = {'BUS': 'BEST Bus', 'CENTRAL': 'Central Line', 'HARBOUR': 'Harbour Line', 'WALK': 'Walk'}
    
    steps = []
    current_mode = None
    segment_start = path[0]
    segment_cost = 0
    stops_in_segment = 0
    
    for i, mode in enumerate(mode_path):
        if mode != current_mode:
            if current_mode is not None:
                icon = icons.get(current_mode, '\U0001f698')
                steps.append({
                    'icon': icon,
                    'mode': names.get(current_mode, current_mode),
                    'from': segment_start,
                    'to': path[i],
                    'stops': stops_in_segment,
                    'cost': segment_cost,
                })
            current_mode = mode
            segment_start = path[i]
            segment_cost = 0
            stops_in_segment = 0
        
        # Estimate segment cost from edge
        for neighbor, edge_cost, m in graph[path[i]]:
            if neighbor == path[i + 1] and m in (mode, 'TRANSFER'):
                segment_cost += edge_cost
                break
        stops_in_segment += 1
    
    # Final segment
    if current_mode:
        icon = icons.get(current_mode, '\U0001f698')
        steps.append({
            'icon': icon,
            'mode': names.get(current_mode, current_mode),
            'from': segment_start,
            'to': path[-1],
            'stops': stops_in_segment,
            'cost': segment_cost,
        })
    
    return steps


def navigate(origin, destination, max_modes=3, top_k=3):
    """Main AI navigation function."""
    # Fuzzy match input names
    src = fuzzy_find_stop(origin, all_nodes)
    dst = fuzzy_find_stop(destination, all_nodes)
    
    if not src:
        print(f"\u274c Origin '{origin}' not found in graph.")
        return
    if not dst:
        print(f"\u274c Destination '{destination}' not found in graph.")
        return
    
    if src != origin.strip().upper():
        print(f"\U0001f50d Matched origin: '{origin}' \u2192 {src}")
    if dst != destination.strip().upper():
        print(f"\U0001f50d Matched destination: '{destination}' \u2192 {dst}")
    
    print(f"\n{'='*70}")
    print(f"\U0001f5fa\ufe0f  NAVIGATION: {src} \u2192 {dst}")
    print(f"   Max modes allowed: {max_modes}")
    print(f"{'='*70}")
    
    # Find top-K routes
    routes = find_top_k_routes(graph, src, dst, max_modes=max_modes, k=top_k)
    
    if not routes:
        # Try single best
        result = dijkstra_multi_modal(graph, src, dst, max_modes=max_modes)
        if result:
            routes = [result]
        else:
            print(f"\n\u26a0\ufe0f No path found within {max_modes} mode limit.")
            return
    
    results_data = []
    for rank, (cost, path, mode_path) in enumerate(routes, 1):
        modes_used = set(m for m in mode_path if m not in ('WALK', 'TRANSFER'))
        steps = format_navigation(cost, path, mode_path)
        
        print(f"\n{'\u2500'*70}")
        tag = '\u2b50 BEST' if rank == 1 else f'#{rank}'
        print(f"{tag} | Cost: \u20b9{cost} | Modes: {len(modes_used)} ({', '.join(sorted(modes_used))}) | Stops: {len(path)}")
        print(f"{'\u2500'*70}")
        
        for step_num, step in enumerate(steps, 1):
            print(f"   {step['icon']} Step {step_num}: Take {step['mode']}")
            print(f"      {step['from']} \u2192 {step['to']} ({step['stops']} stops, \u20b9{step['cost']})")
        
        results_data.append({
            'Route': rank,
            'Cost': f'\u20b9{cost}',
            'Modes': len(modes_used),
            'Mode_Types': ', '.join(sorted(modes_used)),
            'Total_Stops': len(path),
            'Segments': len(steps),
        })
    
    print(f"\n")
    display(pd.DataFrame(results_data))
    return routes

print("\u2705 AI Route Navigator ready!")
print("\nUsage: navigate('CST', 'Ghatkopar', max_modes=3, top_k=3)")

# COMMAND ----------

# DBTITLE 1,Demo - Multi-Modal Navigation
# ============================================================
# DEMO: Navigate across Mumbai using multiple transport modes
# ============================================================

# === CHANGE THESE TO TEST ===
ORIGIN = "Churchgate"
DESTINATION = "Thane"
MAX_MODES = 3
# ============================

print("\U0001f30d MULTI-MODAL ROUTE FINDER")
print(f"   From: {ORIGIN}")
print(f"   To: {DESTINATION}")
print(f"   Max modes: {MAX_MODES}\n")

routes = navigate(ORIGIN, DESTINATION, max_modes=MAX_MODES, top_k=3)

# Also test with restricted modes
if routes:
    print(f"\n\n{'='*70}")
    print("\U0001f50d WHAT IF: Only 1 mode allowed?")
    print(f"{'='*70}")
    navigate(ORIGIN, DESTINATION, max_modes=1, top_k=1)