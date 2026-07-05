import streamlit as st
import requests
import pandas as pd

API_URL = "https://vct-predictor.onrender.com"

st.set_page_config(page_title="VCT 2023 Match Predictor", page_icon="🎯", layout="wide")

st.title("🎯 VCT 2023 Match Predictor")
st.caption("AIT403 Advanced Data Analysis — Powered by XGBoost + FastAPI")

# ── Load team list from artifacts ────────────────────────────────────────────
@st.cache_data
def load_teams():
    try:
        df = pd.read_csv("model_artifacts/team_avg_stats.csv")
        return sorted(df["team"].dropna().unique().tolist())
    except:
        return ["LOUD","NRG","Fnatic","Cloud9","DRX","Paper Rex","ZETA Division","T1"]

@st.cache_data
def load_map_wr():
    try:
        return pd.read_csv("model_artifacts/map_wr_lookup.csv")
    except:
        return pd.DataFrame()

teams    = load_teams()
map_wr   = load_map_wr()
ALL_MAPS = ["Ascent","Pearl","Haven","Lotus","Fracture","Bind","Split"]

# ── Sidebar config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Match settings")
    fmt = st.radio("Series format", ["BO1","BO3","BO5"],
                   captions=["1 map","Best of 3","Best of 5"])
    st.divider()
    st.subheader("Team A")
    team_a = st.selectbox("Select team A", teams, index=0, key="ta")
    st.subheader("Team B")
    remaining = [t for t in teams if t != team_a]
    team_b = st.selectbox("Select team B", remaining, index=0, key="tb")
    st.divider()
    predict_btn = st.button("🔮 Predict series", use_container_width=True, type="primary")

# ── Map win rate helper ───────────────────────────────────────────────────────
def get_wr(team, map_name):
    if map_wr.empty:
        return 0.5
    r = map_wr[(map_wr["team"]==team) & (map_wr["map"]==map_name)]
    return float(r["win_rate"].values[0]) if not r.empty else 0.5

# ── Veto simulation ───────────────────────────────────────────────────────────
def simulate_veto(ta, tb, fmt):
    pool  = ALL_MAPS.copy()
    used, played, log = [], [], []

    def ban(team, name):
        worst = sorted([m for m in pool if m not in used], key=lambda m: get_wr(team,m))[0]
        used.append(worst)
        log.append({"team":name,"action":"Ban","map":worst})

    def pick(team, name, tag):
        best = sorted([m for m in pool if m not in used], key=lambda m: get_wr(team,m), reverse=True)[0]
        used.append(best)
        played.append({"map":best,"type":tag})
        log.append({"team":name,"action":"Pick","map":best})

    def decider():
        rem = [m for m in pool if m not in used][0]
        used.append(rem)
        played.append({"map":rem,"type":"Decider"})
        log.append({"team":"–","action":"Decider","map":rem})

    if fmt == "BO1":
        for _ in range(3): ban(ta,"Team A"); ban(tb,"Team B")
        decider()
    elif fmt == "BO3":
        ban(ta,"Team A"); ban(tb,"Team B")
        ban(ta,"Team A"); ban(tb,"Team B")
        pick(ta,"Team A","Team A pick"); pick(tb,"Team B","Team B pick"); decider()
    else:
        ban(ta,"Team A"); ban(tb,"Team B")
        pick(ta,"Team A","Team A pick"); pick(ta,"Team A","Team A pick")
        pick(tb,"Team B","Team B pick"); pick(tb,"Team B","Team B pick"); decider()

    return played, log

# ── Main content ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2,1,2])
with col1:
    st.metric("Team A", team_a)
with col2:
    st.markdown("<h3 style='text-align:center;margin-top:8px'>vs</h3>", unsafe_allow_html=True)
with col3:
    st.metric("Team B", team_b)

# Map win rate comparison
st.subheader("📊 Map win rates")
wr_data = {"Map": ALL_MAPS,
           team_a: [round(get_wr(team_a,m)*100,1) for m in ALL_MAPS],
           team_b: [round(get_wr(team_b,m)*100,1) for m in ALL_MAPS]}
wr_df = pd.DataFrame(wr_data).set_index("Map")
st.dataframe(wr_df.style.highlight_max(axis=1, color="#d4edda")
                        .format("{:.1f}%"), use_container_width=True)

# Veto preview
st.subheader("🗺️ Map veto simulation")
played, log = simulate_veto(team_a, team_b, fmt)

log_df = pd.DataFrame(log)
log_df.index = range(1, len(log_df)+1)
log_df.columns = ["Team","Action","Map"]

col_log, col_maps = st.columns([1,1])
with col_log:
    st.caption("Veto sequence")
    st.dataframe(log_df, use_container_width=True)
with col_maps:
    st.caption("Maps to be played")
    for i, v in enumerate(played):
        tag_color = {"Team A pick":"🔵","Team B pick":"🔴","Decider":"🟡"}.get(v["type"],"⚪")
        st.markdown(f"**Map {i+1}:** {tag_color} {v['map']} — *{v['type']}*")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    st.divider()
    st.subheader("🏆 Series prediction")

    max_wins  = 3 if fmt=="BO5" else 2 if fmt=="BO3" else 1
    score_a, score_b = 0, 0
    map_results = []

    with st.spinner("Calling FastAPI model..."):
        for v in played:
            if score_a >= max_wins or score_b >= max_wins:
                break
            try:
                resp = requests.post(f"{API_URL}/predict",
                                     json={"team_a":team_a,"team_b":team_b,
                                           "map_name":v["map"]}, timeout=10)
                if resp.status_code == 200:
                    r = resp.json()
                    map_results.append({**v, **r})
                    if r["predicted_winner"] == team_a: score_a += 1
                    else: score_b += 1
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
                    st.stop()
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to FastAPI. Make sure `uvicorn app:app --reload --port 8000` is running.")
                st.stop()

    # Series result
    winner = team_a if score_a >= max_wins else team_b
    c1, c2, c3 = st.columns([2,1,2])
    with c1:
        st.metric(team_a, score_a, delta="Winner ✓" if winner==team_a else None)
    with c2:
        st.markdown("<h2 style='text-align:center'>–</h2>", unsafe_allow_html=True)
    with c3:
        st.metric(team_b, score_b, delta="Winner ✓" if winner==team_b else None)

    st.success(f"🏆 **{winner}** predicted to win the {fmt} series {score_a}–{score_b}")

    # Per-map results
    st.subheader("Per-map breakdown")
    for i, r in enumerate(map_results):
        prob_a = r.get("team_a_win_prob", 0.5)
        prob_b = r.get("team_b_win_prob", 0.5)
        conf   = r.get("confidence","–")
        winner_map = r.get("predicted_winner","–")
        conf_color = {"High":"🟢","Medium":"🟡","Low":"🔴"}.get(conf,"⚪")

        with st.expander(f"Map {i+1} — {r['map']} ({r['type']}) → {winner_map} wins {conf_color}"):
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric(f"{team_a} win prob", f"{prob_a*100:.1f}%")
            with mc2:
                st.metric(f"{team_b} win prob", f"{prob_b*100:.1f}%")
            st.progress(prob_a)
            st.caption(f"Confidence: {conf}")

    # AI insight
    st.subheader("🤖 AI agent insight")
    maps_won_a = [r["map"] for r in map_results if r.get("predicted_winner")==team_a]
    maps_won_b = [r["map"] for r in map_results if r.get("predicted_winner")==team_b]
    best_map   = max(map_results, key=lambda r: max(r.get("team_a_win_prob",0.5),
                                                     r.get("team_b_win_prob",0.5)))
    insight = (
        f"{winner} is predicted to win the {fmt} series {score_a}–{score_b}. "
        f"{team_a} expected to take {', '.join(maps_won_a) if maps_won_a else 'no maps'}; "
        f"{team_b} expected to take {', '.join(maps_won_b) if maps_won_b else 'no maps'}. "
        f"Highest-confidence map: **{best_map['map']}** at "
        f"{max(best_map.get('team_a_win_prob',0.5), best_map.get('team_b_win_prob',0.5))*100:.0f}%."
    )
    st.info(insight)

    # Raw JSON for report
    with st.expander("📋 Raw API response (for report)"):
        st.json(map_results)