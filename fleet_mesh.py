"""
Fleet Mesh — Streamlit
Safe and Efficient Operation of Mine Vehicles in Fog and Low-Visibility Conditions

GitHub → Streamlit Cloud:
  repo root must contain: fleet_mesh.py, requirements.txt, ore_mine_map.jpg
  Main file path: fleet_mesh.py
"""

from __future__ import annotations

import math
import random
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

MAP_W, MAP_H = 1290, 894
MAP_PATH = Path(__file__).parent / "ore_mine_map.jpg"
PX_PER_M = 0.25  # 4 m per pixel

# --- Live update / mesh timing (tune these to speed up or slow down the demo) ---
TICK_SECONDS = 0.5          # how often the live panel refreshes (was 2.0)
DROPOUT_MIN = 0.4           # min simulated "packet in flight" / dropout time (was 2.0)
DROPOUT_RANGE = 1.0         # added random range on top of DROPOUT_MIN (was 4.0)
STATUS_LIVE_S = 1.2         # age <= this => "live" (was 5)
STATUS_CAUTION_S = 2.4      # age <= this => "caution" (was 8)

ROADS = {
    "topTraverse": [
        [0, 95], [100, 80], [200, 60], [300, 75], [390, 95], [480, 80], [560, 55],
        [610, 60], [650, 90], [700, 95], [790, 115], [900, 110], [1010, 55],
        [1100, 60], [1200, 70], [1290, 80],
    ],
    "centralDescent": [
        [600, 90], [590, 150], [570, 210], [555, 260], [520, 300], [500, 340],
        [490, 380], [470, 420], [460, 460], [450, 500], [440, 540], [460, 570],
        [490, 600], [520, 630], [540, 660], [520, 700], [480, 730], [440, 760],
        [420, 800], [410, 860],
    ],
    "eastBranch": [
        [440, 540], [460, 570], [490, 600], [540, 620], [600, 635], [650, 650],
        [700, 645], [750, 620], [800, 600], [850, 580], [900, 555], [950, 545],
    ],
    "westLoop": [[10, 220], [100, 215], [160, 255], [100, 310], [20, 335], [10, 300], [10, 220]],
    "farWestVertical": [
        [80, 340], [85, 420], [75, 500], [70, 580], [65, 660], [75, 740], [80, 820], [85, 894],
    ],
    "eastSettlement": [
        [900, 110], [950, 180], [1000, 230], [1030, 280], [1010, 340], [970, 400],
        [950, 460], [960, 520], [1000, 560], [1050, 600],
    ],
}

TRUCK_DEFS = [
    {"id": "TRK-01", "role": "Haul A", "payload": "Iron ore · rim road", "road": "topTraverse", "v_kph": 22.0, "frac": 0.40},
    {"id": "TRK-02", "role": "Haul B", "payload": "Iron ore (empty)", "road": "topTraverse", "v_kph": 26.0, "frac": 0.44},
    {"id": "TRK-03", "role": "Haul C", "payload": "Waste rock · rim", "road": "topTraverse", "v_kph": 20.0, "frac": 0.48},
    {"id": "TRK-04", "role": "Haul D", "payload": "Iron ore · pit descent", "road": "centralDescent", "v_kph": 16.0, "frac": 0.35},
    {"id": "TRK-05", "role": "Water cart", "payload": "Dust suppressant", "road": "centralDescent", "v_kph": 18.0, "frac": 0.39},
    {"id": "TRK-06", "role": "Service", "payload": "Fuel / parts", "road": "centralDescent", "v_kph": 12.0, "frac": 0.43},
]

LIVE, CAUTION, STALE = "#3ecf6e", "#f5a623", "#e5484d"


def path_len(pts):
    return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts)))


def point_at(pts, s):
    L = path_len(pts)
    if L <= 0:
        return pts[0][0], pts[0][1]
    s = max(0.0, min(L, s))
    acc = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        seg = math.hypot(dx, dy)
        if acc + seg >= s or i == len(pts) - 1:
            f = 0 if seg <= 0 else (s - acc) / seg
            f = max(0.0, min(1.0, f))
            return pts[i - 1][0] + dx * f, pts[i - 1][1] + dy * f
        acc += seg
    return pts[-1][0], pts[-1][1]


def init_state():
    if "trucks" in st.session_state:
        return
    trucks = []
    for d in TRUCK_DEFS:
        pts = ROADS[d["road"]]
        L = path_len(pts)
        s = d["frac"] * L
        x, y = point_at(pts, s)
        trucks.append(
            {
                **d,
                "pts": pts,
                "L": L,
                "s": s,
                "dir": 1,
                "x": x,
                "y": y,
                "heading": 0.0,
                "speed": d["v_kph"],
                "hops": 1,
                "age": 0.0,
                "status": "live",
                "dropout": 0.0,
                "trail": [(x, y)],
            }
        )
    st.session_state.trucks = trucks
    st.session_state.pkt = 0


def step(dt: float):
    for tr in st.session_state.trucks:
        if tr["dropout"] > 0:
            tr["dropout"] -= dt
            tr["age"] += dt
        else:
            if random.random() < 0.02:
                tr["dropout"] = DROPOUT_MIN + random.random() * DROPOUT_RANGE
            step_px = (tr["v_kph"] / 3.6) * dt * PX_PER_M
            news = tr["s"] + tr["dir"] * step_px
            if news >= tr["L"]:
                news = tr["L"]
                tr["dir"] = -1
            if news <= 0:
                news = 0
                tr["dir"] = 1
            nx, ny = point_at(tr["pts"], news)
            dist = math.hypot(nx - tr["x"], ny - tr["y"])
            if dist > 0.15:
                tr["heading"] = (math.degrees(math.atan2(ny - tr["y"], nx - tr["x"])) + 360) % 360
            spd = (dist / PX_PER_M / max(dt, 0.05)) * 3.6
            tr.update(
                s=news,
                x=nx,
                y=ny,
                speed=min(28.0, max(8.0, spd)),
                hops=1 + random.randint(0, 2),
                age=0.0,
            )
            tr["trail"].append((nx, ny))
            tr["trail"] = tr["trail"][-40:]
            st.session_state.pkt += 1
        tr["status"] = "live" if tr["age"] <= STATUS_LIVE_S else ("caution" if tr["age"] <= STATUS_CAUTION_S else "stale")


def build_fig(fog: bool, show_links: bool, show_roads: bool):
    fig = go.Figure()
    img = Image.open(MAP_PATH)
    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=MAP_W,
            sizey=MAP_H,
            sizing="stretch",
            opacity=0.55 if fog else 1.0,
            layer="below",
        )
    )
    if show_roads:
        first = True
        for name, pts in ROADS.items():
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in pts],
                    y=[p[1] for p in pts],
                    mode="lines",
                    line=dict(color="rgba(62,207,110,0.55)", width=2),
                    name="Haul road",
                    showlegend=first,
                    hovertemplate=name + "<extra></extra>",
                )
            )
            first = False

    fig.add_trace(
        go.Scatter(
            x=[580],
            y=[535],
            mode="markers+text",
            marker=dict(size=14, color="#e5484d", symbol="circle-open", line=dict(width=2)),
            text=["REF"],
            textposition="top center",
            textfont=dict(color="#ffb3b3", size=11),
            name="Site datum",
            hovertemplate="Site datum<extra></extra>",
        )
    )

    trucks = st.session_state.trucks
    if show_links:
        live = [t for t in trucks if t["status"] != "stale"]
        for i, a in enumerate(live):
            for b in live[i + 1 :]:
                if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 380:
                    fig.add_trace(
                        go.Scatter(
                            x=[a["x"], b["x"]],
                            y=[a["y"], b["y"]],
                            mode="lines",
                            line=dict(color="rgba(62,207,110,0.4)", width=1),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
    for tr in trucks:
        col = {"live": LIVE, "caution": CAUTION, "stale": STALE}[tr["status"]]
        if len(tr["trail"]) > 1:
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in tr["trail"]],
                    y=[p[1] for p in tr["trail"]],
                    mode="lines",
                    line=dict(color=col, width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[tr["x"]],
                y=[tr["y"]],
                mode="markers+text",
                marker=dict(size=16, color=col, symbol="diamond", line=dict(color="#0d0f13", width=1)),
                text=[tr["id"]],
                textposition="bottom center",
                textfont=dict(color="#e9edf1", size=11),
                name=tr["id"],
                hovertemplate=(
                    f"<b>{tr['id']}</b><br>{tr['role']}<br>{tr['payload']}<br>"
                    f"{tr['speed']:.0f} km/h · hdg {tr['heading']:.0f}°<br>"
                    f"hops {tr['hops']}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        paper_bgcolor="#12151a",
        plot_bgcolor="#1a1612",
        margin=dict(l=10, r=10, t=36, b=10),
        height=620,
        legend=dict(font=dict(color="#8b95a1", size=11), bgcolor="rgba(18,21,26,0.75)"),
        title=dict(
            text="Ore mine haul network · LoRa P2P mesh · fog / low vis",
            font=dict(color="#e9edf1", size=14),
        ),
        xaxis=dict(visible=False, range=[0, MAP_W], constrain="domain"),
        yaxis=dict(visible=False, range=[MAP_H, 0], scaleanchor="x", scaleratio=1),
        font=dict(color="#8b95a1"),
    )
    return fig


def render(fog, show_links, show_roads):
    trucks = st.session_state.trucks
    live_n = sum(1 for t in trucks if t["status"] == "live")
    stale_n = sum(1 for t in trucks if t["status"] == "stale")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mesh nodes", len(trucks))
    c2.metric("Live", live_n)
    c3.metric("Stale", stale_n)
    c4.metric("Packets", st.session_state.pkt)
    st.plotly_chart(build_fig(fog, show_links, show_roads), use_container_width=True)
    rows = [
        {
            "Truck": t["id"],
            "Role": t["role"],
            "Payload": t["payload"],
            "Status": t["status"].upper(),
            "Speed km/h": round(t["speed"]),
            "Heading": round(t["heading"]),
            "Seen s": int(t["age"]),
            "Hops": t["hops"],
        }
        for t in trucks
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="Fleet Mesh — Ore Mine", page_icon="🚛", layout="wide")
    st.markdown(
        """
        <style>
          .stApp { background:#12151a; color:#8b95a1; }
          [data-testid="stSidebar"] { background:#1a1f26; }
          h1,h2,h3 { color:#e9edf1 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if not MAP_PATH.exists():
        st.error("ore_mine_map.jpg is missing. Put it next to fleet_mesh.py in the GitHub repo.")
        st.stop()
    init_state()

    st.title("FLEET MESH")
    st.caption(
        "Safe and Efficient Operation of Mine Vehicles in Fog and Low-Visibility Conditions  ·  "
        "LoRa P2P  ·  trucks on surveyed haul roads"
    )

    fog = st.sidebar.toggle("Fog / low visibility", value=True)
    show_links = st.sidebar.toggle("LoRa mesh links", value=True)
    show_roads = st.sidebar.toggle("Show haul centrelines", value=True)
    auto = st.sidebar.toggle("Live demo (fast tick)", value=True)
    st.sidebar.markdown(
        """
        **Terrain (from map)**  
        Green lines = haul roads  
        Trucks 12–28 km/h  
        REF = site datum
        """
    )
    if st.sidebar.button("Step once"):
        step(1.0)

    if auto:

        @st.fragment(run_every=timedelta(seconds=TICK_SECONDS))
        def live_panel():
            step(TICK_SECONDS)
            render(fog, show_links, show_roads)

        live_panel()
    else:
        render(fog, show_links, show_roads)


main()
