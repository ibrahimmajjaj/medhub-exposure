#!/usr/bin/env python3
"""
hub_exposure.py — Geodata-driven systemic exposure analysis of a freight hub's
hinterland network (MedHub–Tanger Med case).

Pipeline (mirrors Sections 3-5 of the manuscript):
  1. Build a multimodal (road + rail) freight network from OpenStreetMap.
  2. Snap the hub and hinterland freight nodes to the network.
  3. Compute baseline generalized travel times from the hub to every destination.
  4. Single-link removal: link importance (Jenelius et al. 2006), unsatisfied
     demand, destination exposure, and redundancy (edge-disjoint paths).
  5. Compound "poly-crisis" scenarios (rail-layer outage, regional closure,
     rail outage + worst road link).
  6. Governance mediation: disruption-hours under coordinated vs fragmented
     governance regimes (duration and procedural-delay parameters).
  7. Exposure-absorption typology of hinterland destinations.

Usage
  python hub_exposure.py --config config.json --out results/
  python hub_exposure.py --synthetic --out test_results/     # self-test, no internet

Requirements: osmnx>=2.0, networkx>=3, geopandas, pandas, numpy, matplotlib.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROAD_FILTER = ('["highway"~"motorway|motorway_link|trunk|trunk_link|primary|'
               'primary_link|secondary|secondary_link"]')
RAIL_FILTER = '["railway"~"rail"]["service"!~"yard|siding|spur"]'


# --------------------------------------------------------------------------- #
# 1. Network construction
# --------------------------------------------------------------------------- #
def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _to_simple_undirected(G_osm, mode: str, speeds_kmh: dict, prefix: str) -> nx.Graph:
    """Collapse an OSMnx MultiDiGraph into an undirected simple graph whose edge
    weight 'tt_h' is the fastest travel time (hours) between two nodes."""
    H = nx.Graph()
    for n, d in G_osm.nodes(data=True):
        H.add_node(f"{prefix}{n}", x=d["x"], y=d["y"], mode=mode)
    for u, v, d in G_osm.edges(data=True):
        hw = d.get("highway", "rail")
        hw = hw[0] if isinstance(hw, list) else hw
        speed = speeds_kmh.get(hw, speeds_kmh.get("default", 50.0))
        tt = (d.get("length", 0.0) / 1000.0) / speed
        a, b = f"{prefix}{u}", f"{prefix}{v}"
        if a == b:
            continue
        if not H.has_edge(a, b) or H[a][b]["tt_h"] > tt:
            H.add_edge(a, b, tt_h=tt, length_m=d.get("length", 0.0), mode=mode,
                       hwy=hw, osmid=str(d.get("osmid", "")),
                       name=_label(d))
    return H


def _label(d):
    def j(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return ""
        return " / ".join(map(str, x)) if isinstance(x, list) else str(x)
    ref, name = j(d.get("ref")), j(d.get("name"))
    return (f"{ref} {name}".strip()) if ref else name


def nearest_node(G: nx.Graph, lat: float, lon: float, mode: str | None = None):
    nodes = [n for n, d in G.nodes(data=True) if mode is None or d["mode"] == mode]
    ys = np.array([G.nodes[n]["y"] for n in nodes])
    xs = np.array([G.nodes[n]["x"] for n in nodes])
    dist = _haversine_km(lat, lon, ys, xs)
    i = int(np.argmin(dist))
    return nodes[i], float(dist[i])


EXCLUDED_RAIL_SERVICE = {"yard", "siding", "spur", "crossover"}


def _drop_service_rail(G_osm):
    drop = []
    for u, v, k, d in G_osm.edges(keys=True, data=True):
        s = d.get("service")
        s = s if isinstance(s, list) else [s]
        if any(x in EXCLUDED_RAIL_SERVICE for x in s if x):
            drop.append((u, v, k))
    G_osm.remove_edges_from(drop)
    G_osm.remove_nodes_from([n for n in list(G_osm.nodes) if G_osm.degree(n) == 0])
    return G_osm


def build_network(cfg: dict) -> nx.Graph:
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.log_console = False
    if cfg.get("road_xml"):
        # Offline mode: OSM XML files pre-filtered with osmium from a Geofabrik extract
        road = ox.graph_from_xml(cfg["road_xml"], retain_all=False)
    else:
        bbox = tuple(cfg["bbox"])  # (west, south, east, north)
        road = ox.graph_from_bbox(bbox, custom_filter=ROAD_FILTER, retain_all=False)
    G = _to_simple_undirected(road, "road", cfg["road_speeds_kmh"], "r")
    if cfg.get("include_rail", True):
        if cfg.get("rail_xml"):
            # keep service tracks (yards, sidings, spurs): they are physical connectors to
            # plants and port terminals; they receive a low speed instead of being dropped
            rail = ox.graph_from_xml(cfg["rail_xml"], retain_all=True)
        else:
            rail = ox.graph_from_bbox(tuple(cfg["bbox"]), custom_filter='["railway"~"rail"]', retain_all=True)
        for u, v, k, d in rail.edges(keys=True, data=True):
            sv = d.get("service")
            sv = sv if isinstance(sv, list) else [sv]
            d["highway"] = "rail_service" if any(x in EXCLUDED_RAIL_SERVICE for x in sv if x) else "rail"
        R = _to_simple_undirected(rail, "rail",
                                  {"rail": cfg["rail_speed_kmh"],
                                   "rail_service": cfg.get("rail_service_speed_kmh", 20),
                                   "default": cfg["rail_speed_kmh"]}, "t")
        G = nx.compose(G, R)
        # Intermodal transfer edges at declared rail-road terminals
        for term in cfg.get("terminals", []):
            rn, dr = nearest_node(G, term["lat"], term["lon"], "road")
            tn, dt = nearest_node(G, term["lat"], term["lon"], "rail")
            if max(dr, dt) <= cfg.get("max_snap_km", 3.0):
                G.add_edge(rn, tn, tt_h=cfg["transfer_h"], length_m=0.0,
                           mode="transfer", hwy="transfer", osmid="", name=term["name"])
        G.graph["transfers"] = [d["name"] for *_, d in G.edges(data=True) if d.get("mode") == "transfer"]
    # keep the largest connected component of the road layer (+ everything attached
    # to it) and ALL rail components, so that rail-only relations can be evaluated
    giant = max(nx.connected_components(G), key=len)
    keep = set(giant) | {n for n, d in G.nodes(data=True) if d["mode"] == "rail"}
    H = G.subgraph(keep).copy()
    H.graph.update(G.graph)
    return H


# --------------------------------------------------------------------------- #
# 2-3. Freight nodes, OD relations and baseline
# --------------------------------------------------------------------------- #
def snap_points(G, points: list[dict], max_snap_km: float, mode="road") -> pd.DataFrame:
    rows = []
    for p in points:
        n, d = nearest_node(G, p["lat"], p["lon"], mode)
        rows.append({**p, "node": n, "snap_km": d, "snapped_ok": d <= max_snap_km})
    return pd.DataFrame(rows)


class Ctx:
    """Hub and OD relations. A relation is either road-based (multimodal shortest
    path from the hub road node) or rail-captive (rail-only path from the hub rail
    node; if rail is unavailable, shipments shift to road with a modal-shift penalty)."""

    def __init__(self, G, cfg, dests: pd.DataFrame):
        self.hub_road = nearest_node(G, cfg["hub"]["lat"], cfg["hub"]["lon"], "road")[0]
        self.hub_rail = None
        self.modal_shift_h = cfg.get("modal_shift_h", 12.0)
        self.d = dests.reset_index(drop=True).copy()
        self.d["w"] = self.d["weight"] / self.d["weight"].sum()
        self.d["captive"] = self.d.get("rail_captive", False).fillna(False).astype(bool) \
            if "rail_captive" in self.d else False
        if self.d["captive"].any():
            hr = cfg.get("hub_rail", cfg["hub"])
            self.hub_rail, self.hub_rail_km = nearest_node(G, hr["lat"], hr["lon"], "rail")
            rn = [nearest_node(G, r.lat, r.lon, "rail") if r.captive else (None, np.nan)
                  for r in self.d.itertuples()]
            self.d["rail_node"] = [x[0] for x in rn]
            self.d["rail_snap_km"] = [x[1] for x in rn]
        else:
            self.d["rail_node"] = None
        self.keys = list(range(len(self.d)))


def _rail_view(H):
    return nx.subgraph_view(H, filter_edge=lambda u, v: H[u][v]["mode"] == "rail")


def od_costs(H, ctx: Ctx, hub_delay_h=0.0, want_paths=False):
    """Return {relation index: cost}, {index: shifted?}, optional {index: path}."""
    dist, paths = nx.single_source_dijkstra(H, ctx.hub_road, weight="tt_h")
    dr, pr = ({}, {})
    if ctx.hub_rail is not None and ctx.hub_rail in H:
        dr, pr = nx.single_source_dijkstra(_rail_view(H), ctx.hub_rail, weight="tt_h")
    cost, shifted, pth = {}, {}, {}
    for i, r in ctx.d.iterrows():
        if r["captive"] and r["rail_node"] in dr:
            cost[i], shifted[i], pth[i] = dr[r["rail_node"]], False, pr[r["rail_node"]]
        elif r["node"] in dist:
            extra = ctx.modal_shift_h if r["captive"] else 0.0
            cost[i], shifted[i], pth[i] = dist[r["node"]] + extra, bool(r["captive"]), paths[r["node"]]
        else:
            cost[i], shifted[i], pth[i] = math.inf, False, None
        if not math.isinf(cost[i]):
            cost[i] += hub_delay_h
    return (cost, shifted, pth) if want_paths else (cost, shifted)


# --------------------------------------------------------------------------- #
# 4. Single-link removal (Algorithm 1 in the manuscript)
# --------------------------------------------------------------------------- #
def link_removal(G, ctx: Ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = ctx.d["w"].to_dict()
    c0, _, paths = od_costs(G, ctx, want_paths=True)
    # Only links on at least one baseline OD path can change OD costs under
    # all-or-nothing shortest-path assignment -> candidate set.
    cand = {}
    for i, p in paths.items():
        if p is None:
            continue
        for a, b in zip(p[:-1], p[1:]):
            cand.setdefault(tuple(sorted((a, b))), []).append(i)
    edge_rows, delta = [], {i: [] for i in ctx.keys}
    for (a, b), users in cand.items():
        attrs = dict(G[a][b])
        G.remove_edge(a, b)
        c1, sh = od_costs(G, ctx)
        G.add_edge(a, b, **attrs)
        imp, unsat, worst, worst_d, nshift = 0.0, 0.0, 0.0, None, 0
        for i in users:
            if math.isinf(c1[i]):
                unsat += w[i]; delta[i].append(math.inf); worst, worst_d = math.inf, i
            else:
                inc = c1[i] - c0[i]
                imp += w[i] * inc; delta[i].append(inc); nshift += int(sh[i])
                if inc > worst:
                    worst, worst_d = inc, i
        edge_rows.append({"u": a, "v": b, "mode": attrs["mode"], "hwy": attrs["hwy"],
                          "name": attrs["name"], "length_km": attrs["length_m"] / 1000,
                          "lat": (G.nodes[a]["y"] + G.nodes[b]["y"]) / 2,
                          "lon": (G.nodes[a]["x"] + G.nodes[b]["x"]) / 2,
                          "n_od_users": len(users),
                          "importance_h": imp, "unsatisfied_share": unsat,
                          "worst_delay_h": worst,
                          "worst_dest": ctx.d.loc[worst_d, "name"] if worst_d is not None else None,
                          "modal_shifts": nshift})
    edges = pd.DataFrame(edge_rows).sort_values(
        ["unsatisfied_share", "importance_h"], ascending=False).reset_index(drop=True)
    rv = nx.Graph(_rail_view(G))
    dest_rows = []
    for i, r in ctx.d.iterrows():
        ds = delta[i]
        exp_h = max(ds) if ds else 0.0
        try:
            k = (nx.edge_connectivity(rv, ctx.hub_rail, r["rail_node"]) if r["captive"]
                 else nx.edge_connectivity(G, ctx.hub_road, r["node"]))
        except (nx.NetworkXError, KeyError):
            k = 0
        dest_rows.append({"name": r["name"], "relation": "rail-captive" if r["captive"] else "road/multimodal",
                          "weight": w[i], "baseline_tt_h": c0[i],
                          "exposure_worst_delay_h": exp_h,
                          "exposure_ratio": (exp_h / c0[i]) if c0[i] > 0 else np.nan,
                          "edge_disjoint_paths": k,
                          "single_point_of_failure": k <= 1})
    return edges, pd.DataFrame(dest_rows)


# --------------------------------------------------------------------------- #
# 5. Compound / poly-crisis scenarios
# --------------------------------------------------------------------------- #
def scenario_cost(G, ctx, remove_edges=(), hub_delay_h=0.0):
    H = G.copy()
    H.remove_edges_from([e for e in remove_edges if H.has_edge(*e)])
    w = ctx.d["w"].to_dict()
    c, sh = od_costs(H, ctx, hub_delay_h)
    c0, _ = od_costs(G, ctx)
    conn = [i for i in ctx.keys if not math.isinf(c[i])]
    return {"delta_connected_h": sum(w[i] * (c[i] - c0[i]) for i in conn),
            "unsatisfied_share": sum(w[i] for i in ctx.keys if math.isinf(c[i])),
            "shifted_share": sum(w[i] for i in conn if sh[i]),
            "per_dest": c}


def run_scenarios(G, ctx, edges_df, cfg) -> pd.DataFrame:
    rail_edges = [(u, v) for u, v, d in G.edges(data=True) if d["mode"] in ("rail", "transfer")]
    road_top = edges_df[edges_df["mode"] == "road"].sort_values(
        ["unsatisfied_share", "importance_h"], ascending=False).head(1)
    worst_road = [(r.u, r.v) for r in road_top.itertuples()]
    scen = {"S0 baseline": dict(remove_edges=()),
            "S1 worst single road link": dict(remove_edges=worst_road),
            "S2 rail layer outage": dict(remove_edges=rail_edges),
            "S3 compound (S1+S2)": dict(remove_edges=worst_road + rail_edges)}
    for reg in cfg.get("regional_closures", []):
        lat, lon, rad = reg["lat"], reg["lon"], reg["radius_km"]
        inside = {n for n, d in G.nodes(data=True)
                  if _haversine_km(lat, lon, d["y"], d["x"]) <= rad}
        inside.discard(ctx.hub_road)
        reg_edges = [(u, v) for u, v in G.edges() if u in inside or v in inside]
        scen[f"S4 regional closure: {reg['name']}"] = dict(remove_edges=reg_edges)
        scen[f"S5 compound (S4 {reg['name']} + S2)"] = dict(remove_edges=reg_edges + rail_edges)
    rows = []
    for name, kw in scen.items():
        r = scenario_cost(G, ctx, **kw)
        rows.append({"scenario": name, "removed_links": len(kw["remove_edges"]),
                     "delta_connected_h": r["delta_connected_h"],
                     "unsatisfied_share": r["unsatisfied_share"],
                     "shifted_share": r["shifted_share"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. Governance mediation
# --------------------------------------------------------------------------- #
def governance_loss(scen_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Disruption-hours index per daily unit of hub demand (Eq. 6 of the manuscript):
        L_g = D_g * [dC_conn + delta_g * (1 - U)] + U * D_g * (24 * D_g / 2)
    D_g: disruption duration (days) under regime g; delta_g: procedural dwell (h);
    dC_conn: demand-weighted detour delay of routable shipments (h);
    U: share of demand cut off, which waits on average half the outage."""
    rows = []
    for g in cfg["governance_regimes"]:
        for r in scen_df.itertuples():
            if r.scenario.startswith("S0"):
                continue
            dur, dlt, U = g["duration_days"], g["procedural_delay_h"], r.unsatisfied_share
            loss = dur * (r.delta_connected_h + dlt * (1 - U)) + U * dur * (24 * dur / 2)
            rows.append({"regime": g["name"], "scenario": r.scenario,
                         "duration_days": dur, "procedural_delay_h": g["procedural_delay_h"],
                         "unsatisfied_share": r.unsatisfied_share,
                         "disruption_hours_index": loss})
    out = pd.DataFrame(rows)
    piv = out.pivot_table(index="scenario", columns="regime",
                          values="disruption_hours_index")
    if piv.shape[1] == 2:
        a, b = piv.columns
        piv["mediation_ratio"] = piv[a] / piv[b]
    return out, piv


# --------------------------------------------------------------------------- #
# 7. Exposure-absorption typology
# --------------------------------------------------------------------------- #
def typology(dest_df: pd.DataFrame, absorption: pd.DataFrame | None) -> pd.DataFrame:
    df = dest_df.copy()
    e = df["exposure_ratio"].replace(np.inf, np.nan)
    cap = np.nanmax(e) if np.isfinite(np.nanmax(e)) else 1.0
    df["exposure_index"] = e.fillna(cap * 1.5)  # cut-offs ranked above all detours
    df["exposure_index"] = (df["exposure_index"] - df["exposure_index"].min()) / \
        (df["exposure_index"].max() - df["exposure_index"].min() + 1e-12)
    if absorption is not None:
        df = df.merge(absorption[["name", "absorption_index"]], on="name", how="left")
    else:
        df["absorption_index"] = np.nan
    hi_e = df["exposure_index"] >= df["exposure_index"].median()
    hi_a = df["absorption_index"] >= df["absorption_index"].median()
    lab = np.select([hi_e & ~hi_a, hi_e & hi_a, ~hi_e & ~hi_a, ~hi_e & hi_a],
                    ["Structurally vulnerable", "Highly exposed but adaptive",
                     "Stable but limited", "Anchored and robust"], "n/a")
    df["typology"] = np.where(df["absorption_index"].isna(), "n/a (no absorption data)", lab)
    df["net_vulnerability"] = df["exposure_index"] * (1 - df["absorption_index"])
    return df


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def export(G, edges, out: Path):
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
        geoms = [LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                             (G.nodes[v]["x"], G.nodes[v]["y"])]) for u, v in zip(edges.u, edges.v)]
        gpd.GeoDataFrame(edges, geometry=geoms, crs="EPSG:4326").to_file(
            out / "critical_links.gpkg", driver="GPKG")
    except Exception as exc:  # optional
        print("GeoPackage export skipped:", exc)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        for u, v, d in G.edges(data=True):
            ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]], [G.nodes[u]["y"], G.nodes[v]["y"]],
                    color="#bbbbbb" if d["mode"] == "road" else "#7a9cc6", lw=0.4, zorder=1)
        e = edges.copy()
        e["score"] = np.where(e.unsatisfied_share > 0, e.importance_h.max() * 2 + 1e-9,
                              e.importance_h)
        vmax = e["score"].max() or 1
        for r in e.itertuples():
            ax.plot([G.nodes[r.u]["x"], G.nodes[r.v]["x"]], [G.nodes[r.u]["y"], G.nodes[r.v]["y"]],
                    color=plt.cm.inferno_r(0.15 + 0.85 * r.score / vmax), lw=2.2, zorder=2)
        ax.set_aspect("equal"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title("Link importance (demand-weighted delay; darkest = cut link)")
        fig.tight_layout(); fig.savefig(out / "fig_link_importance.png", dpi=200)
    except Exception as exc:
        print("Figure skipped:", exc)


def run(G, cfg, out: Path, absorption=None):
    out.mkdir(parents=True, exist_ok=True)
    msk = cfg.get("max_snap_km", 3.0)
    dests = snap_points(G, cfg["destinations"], msk)
    ctx = Ctx(G, cfg, dests)
    # snapping report (hub, destinations, rail nodes, terminals)
    rows = [{"type": "hub", "name": cfg["hub"]["name"], "lat": cfg["hub"]["lat"], "lon": cfg["hub"]["lon"],
             "node": ctx.hub_road,
             "snap_km": _haversine_km(cfg["hub"]["lat"], cfg["hub"]["lon"], G.nodes[ctx.hub_road]["y"], G.nodes[ctx.hub_road]["x"])}]
    if ctx.hub_rail is not None:
        rows.append({"type": "hub rail", "name": cfg["hub"]["name"], "node": ctx.hub_rail, "snap_km": ctx.hub_rail_km})
    for r in ctx.d.itertuples():
        rows.append({"type": "destination", "name": r.name, "lat": r.lat, "lon": r.lon, "node": r.node, "snap_km": r.snap_km})
        if r.captive:
            rows.append({"type": "destination rail", "name": r.name, "node": r.rail_node, "snap_km": r.rail_snap_km})
    snaps = pd.DataFrame(rows)
    snaps["node_lat"] = [G.nodes[n]["y"] for n in snaps["node"]]
    snaps["node_lon"] = [G.nodes[n]["x"] for n in snaps["node"]]
    snaps["ok"] = snaps["snap_km"] <= msk
    snaps.to_csv(out / "table_snapping.csv", index=False)
    if not snaps["ok"].all():
        print("WARNING: points snapped farther than max_snap_km:\n", snaps[~snaps.ok][["type", "name", "snap_km"]])
    edges, dest_df = link_removal(G, ctx)
    scen = run_scenarios(G, ctx, edges, cfg)
    gov_long, gov = governance_loss(scen, cfg)
    typo = typology(dest_df, absorption)
    edges.to_csv(out / "table_link_criticality.csv", index=False)
    typo.to_csv(out / "table_destination_exposure.csv", index=False)
    scen.to_csv(out / "table_scenarios.csv", index=False)
    gov_long.to_csv(out / "table_governance_long.csv", index=False)
    gov.to_csv(out / "table_governance_mediation.csv")
    stats = {"nodes": G.number_of_nodes(), "links": G.number_of_edges(),
             "road_km": sum(d["length_m"] for *_, d in G.edges(data=True) if d["mode"] == "road") / 1000,
             "rail_km": sum(d["length_m"] for *_, d in G.edges(data=True) if d["mode"] == "rail") / 1000,
             "transfer_links": [d["name"] for *_, d in G.edges(data=True) if d["mode"] == "transfer"],
             "candidate_links": len(edges),
             "candidate_rail_links": int((edges["mode"] == "rail").sum()),
             "cut_links": int((edges.unsatisfied_share > 0).sum())}
    (out / "network_summary.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    export(G, edges, out)
    pd.set_option("display.width", 250)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(snaps.to_string())
    print(edges.head(15).to_string())
    print(typo.to_string())
    print(scen.to_string())
    print(gov.to_string())


# --------------------------------------------------------------------------- #
# Synthetic self-test (no internet needed)
# --------------------------------------------------------------------------- #
def synthetic():
    rng = np.random.default_rng(1)
    G = nx.Graph()
    nx_, ny_ = 12, 12
    for i in range(nx_):
        for j in range(ny_):
            G.add_node(f"r{i}_{j}", x=-6.0 + i * 0.1, y=34.5 + j * 0.1, mode="road")
    for i in range(nx_):
        for j in range(ny_):
            for di, dj in ((1, 0), (0, 1)):
                a, b = f"r{i}_{j}", f"r{i+di}_{j+dj}"
                if b in G and rng.random() > 0.25:
                    L = 11000 * (1 + rng.random() * 0.3)
                    G.add_edge(a, b, tt_h=L / 1000 / 70, length_m=L, mode="road",
                               hwy="primary", osmid="", name=f"{a}-{b}")
    # a bridge-like single access to a peninsula + a rail line
    G.add_node("r_pen", x=-6.2, y=35.6, mode="road")
    G.add_edge("r0_11", "r_pen", tt_h=0.3, length_m=20000, mode="road", hwy="trunk", osmid="", name="peninsula")
    for k in range(10):
        G.add_node(f"t{k}", x=-5.9 + k * 0.1, y=34.55 + k * 0.1, mode="rail")
        if k:
            G.add_edge(f"t{k-1}", f"t{k}", tt_h=14000 / 1000 / 60, length_m=14000,
                       mode="rail", hwy="rail", osmid="", name="railline")
    G.add_edge("r1_0", "t0", tt_h=3, length_m=0, mode="transfer", hwy="transfer", osmid="", name="T0")
    G.add_edge("r10_9", "t9", tt_h=3, length_m=0, mode="transfer", hwy="transfer", osmid="", name="T9")
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    cfg = {"hub": {"name": "Hub", "lat": 35.6, "lon": -4.95},
           "destinations": [
               {"name": "D-pen", "lat": 35.6, "lon": -6.2, "weight": 1},
               {"name": "D-south", "lat": 34.5, "lon": -6.0, "weight": 3},
               {"name": "D-mid", "lat": 35.0, "lon": -5.5, "weight": 2},
               {"name": "D-west", "lat": 35.2, "lon": -5.9, "weight": 2},
               {"name": "D-rail", "lat": 35.45, "lon": -5.05, "weight": 2, "rail_captive": True}],
           "hub_rail": {"lat": 34.55, "lon": -5.9}, "modal_shift_h": 12,
           "max_snap_km": 15,
           "regional_closures": [{"name": "storm", "lat": 35.0, "lon": -5.5, "radius_km": 15}],
           "governance_regimes": [{"name": "fragmented", "duration_days": 5, "procedural_delay_h": 12},
                                  {"name": "coordinated", "duration_days": 1, "procedural_delay_h": 2}]}
    absorption = pd.DataFrame({"name": ["D-pen", "D-south", "D-mid", "D-west"],
                               "absorption_index": [0.2, 0.8, 0.5, 0.3]})
    return G, cfg, absorption


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--absorption", help="CSV with columns name,absorption_index (0-1)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.synthetic:
        G, cfg, absorption = synthetic()
    else:
        cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
        G = build_network(cfg)
        absorption = pd.read_csv(a.absorption) if a.absorption else None
    run(G, cfg, Path(a.out), absorption)


if __name__ == "__main__":
    main()
