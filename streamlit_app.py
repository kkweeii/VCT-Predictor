import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

API_URL     = "https://vct-predictor.onrender.com"
N8N_WEBHOOK = "https://kkweeii.app.n8n.cloud/webhook/retention"  # replace after setting up webhook in n8n

st.set_page_config(page_title="VCT 2023 Match Predictor", page_icon="🎯", layout="wide")

st.title("🎯 VCT 2023 Match Predictor")
st.caption("AIT403 Advanced Data Analysis — XGBoost + FastAPI + n8n Agentic Workflow")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_teams():
    try:
        df = pd.read_csv("model_artifacts/team_avg_stats.csv")
        return sorted(df["team"].dropna().unique().tolist()), df
    except:
        return ["LOUD","NRG Esports","FNATIC","Cloud9","DRX","Paper Rex"], pd.DataFrame()

@st.cache_data
def load_map_wr():
    try:
        return pd.read_csv("model_artifacts/map_wr_lookup.csv")
    except:
        return pd.DataFrame()

teams, team_stats_df = load_teams()
map_wr_df  = load_map_wr()
ALL_MAPS   = ["Ascent","Pearl","Haven","Lotus","Fracture","Bind","Split"]

def get_wr(team, map_name):
    if map_wr_df.empty: return 0.5
    r = map_wr_df[(map_wr_df["team"]==team) & (map_wr_df["map"]==map_name)]
    return float(r["win_rate"].values[0]) if not r.empty else 0.5

def get_team_stats(team):
    if team_stats_df.empty: return {}
    r = team_stats_df[team_stats_df["team"]==team]
    return r.iloc[0].to_dict() if not r.empty else {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    predict_btn = st.button("🔮 Predict & Send to n8n", use_container_width=True, type="primary")
    st.divider()
    st.caption("Dataset: VCT 2023 — Kaggle")
    st.caption("Model: XGBoost Classifier")
    st.caption("Deployed: FastAPI on Render")
    st.caption("Workflow: n8n + Telegram")

# ── Team comparison header ────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2,1,2])
with col1:
    st.markdown(f"### 🔵 {team_a}")
    sa = get_team_stats(team_a)
    if sa:
        st.metric("Avg ACS", f"{sa.get('avg_acs',0):.1f}")
        st.metric("Avg KD",  f"{sa.get('avg_kd_ratio',0):.2f}")
        st.metric("Avg ADR", f"{sa.get('avg_adr',0):.1f}")
with col2:
    st.markdown("<h2 style='text-align:center;margin-top:40px'>VS</h2>", unsafe_allow_html=True)
with col3:
    st.markdown(f"### 🔴 {team_b}")
    sb = get_team_stats(team_b)
    if sb:
        st.metric("Avg ACS", f"{sb.get('avg_acs',0):.1f}")
        st.metric("Avg KD",  f"{sb.get('avg_kd_ratio',0):.2f}")
        st.metric("Avg ADR", f"{sb.get('avg_adr',0):.1f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🗺️ Map Analysis", "📊 Team Stats", "🏆 Prediction"])

with tab1:
    st.subheader("Map win rates comparison")
    wr_a = [get_wr(team_a, m)*100 for m in ALL_MAPS]
    wr_b = [get_wr(team_b, m)*100 for m in ALL_MAPS]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(ALL_MAPS))
    w = 0.35
    bars_a = ax.bar(x - w/2, wr_a, w, label=team_a, color='#185FA5', alpha=0.85)
    bars_b = ax.bar(x + w/2, wr_b, w, label=team_b, color='#993C1D', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_MAPS, fontsize=10)
    ax.set_ylabel("Win rate (%)")
    ax.set_title("Map win rates by team")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for bar in bars_a:
        ax.annotate(f'{bar.get_height():.0f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars_b:
        ax.annotate(f'{bar.get_height():.0f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    col_a, col_b = st.columns(2)
    with col_a:
        best_a  = ALL_MAPS[wr_a.index(max(wr_a))]
        worst_a = ALL_MAPS[wr_a.index(min(wr_a))]
        st.success(f"✅ {team_a} best map: **{best_a}** ({max(wr_a):.0f}%)")
        st.error(f"❌ {team_a} worst map: **{worst_a}** ({min(wr_a):.0f}%)")
    with col_b:
        best_b  = ALL_MAPS[wr_b.index(max(wr_b))]
        worst_b = ALL_MAPS[wr_b.index(min(wr_b))]
        st.success(f"✅ {team_b} best map: **{best_b}** ({max(wr_b):.0f}%)")
        st.error(f"❌ {team_b} worst map: **{worst_b}** ({min(wr_b):.0f}%)")

with tab2:
    st.subheader("Team stats comparison")
    if not team_stats_df.empty:
        stat_cols = [c for c in team_stats_df.columns if c.startswith('avg_')]
        sa_row = team_stats_df[team_stats_df['team']==team_a]
        sb_row = team_stats_df[team_stats_df['team']==team_b]

        if not sa_row.empty and not sb_row.empty:
            compare = pd.DataFrame({
                'Stat':  [c.replace('avg_','').upper() for c in stat_cols],
                team_a:  [round(sa_row.iloc[0][c], 2) for c in stat_cols],
                team_b:  [round(sb_row.iloc[0][c], 2) for c in stat_cols],
            })
            compare['Advantage'] = compare.apply(
                lambda r: f"🔵 {team_a}" if r[team_a] > r[team_b] else f"🔴 {team_b}", axis=1)
            st.dataframe(compare.set_index('Stat'), use_container_width=True)

            key_stats = [s for s in ['avg_acs','avg_kd_ratio','avg_adr','avg_hs_pct','avg_rating']
                         if s in stat_cols]
            if key_stats:
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                vals_a = [sa_row.iloc[0][s] for s in key_stats]
                vals_b = [sb_row.iloc[0][s] for s in key_stats]
                labels = [s.replace('avg_','').upper() for s in key_stats]
                x2 = np.arange(len(labels))
                ax2.bar(x2 - 0.2, vals_a, 0.4, label=team_a, color='#185FA5', alpha=0.85)
                ax2.bar(x2 + 0.2, vals_b, 0.4, label=team_b, color='#993C1D', alpha=0.85)
                ax2.set_xticks(x2)
                ax2.set_xticklabels(labels)
                ax2.set_title("Key stats comparison")
                ax2.legend()
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close()

with tab3:
    if not predict_btn:
        st.info("👈 Select teams and click **Predict & Send to n8n** in the sidebar.")
    else:
        # ── Step 1: Trigger n8n webhook ───────────────────────────────────────
        st.subheader("📡 Sending to n8n workflow...")
        n8n_triggered = False
        try:
            n8n_resp = requests.post(
                N8N_WEBHOOK,
                json={
                    "team_a": team_a,
                    "team_b": team_b,
                    "format": fmt.lower()
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if n8n_resp.status_code in [200, 201]:
                st.success("✅ n8n workflow triggered! Telegram notification will be sent.")
                n8n_triggered = True
            else:
                st.warning(f"⚠️ n8n webhook returned {n8n_resp.status_code} — check webhook URL.")
        except Exception as e:
            st.warning(f"⚠️ Could not reach n8n webhook: {e}")

        st.divider()

        # ── Step 2: Veto simulation ───────────────────────────────────────────
        def simulate_veto(ta, tb, fmt):
            pool = ALL_MAPS.copy()
            used, played, log = [], [], []
            def ban(team, name):
                worst = sorted([m for m in pool if m not in used],
                               key=lambda m: get_wr(team,m))[0]
                used.append(worst)
                log.append({"Team":name,"Action":"Ban","Map":worst})
            def pick(team, name, tag):
                best = sorted([m for m in pool if m not in used],
                              key=lambda m: get_wr(team,m), reverse=True)[0]
                used.append(best); played.append({"map":best,"type":tag})
                log.append({"Team":name,"Action":"Pick","Map":best})
            def decider():
                rem = [m for m in pool if m not in used][0]
                used.append(rem); played.append({"map":rem,"type":"Decider"})
                log.append({"Team":"–","Action":"Decider","Map":rem})
            if fmt=="BO1":
                for _ in range(3): ban(ta,"Team A"); ban(tb,"Team B")
                decider()
            elif fmt=="BO3":
                ban(ta,"Team A"); ban(tb,"Team B")
                ban(ta,"Team A"); ban(tb,"Team B")
                pick(ta,"Team A","Team A pick")
                pick(tb,"Team B","Team B pick")
                decider()
            else:
                ban(ta,"Team A"); ban(tb,"Team B")
                pick(ta,"Team A","Team A pick"); pick(ta,"Team A","Team A pick")
                pick(tb,"Team B","Team B pick"); pick(tb,"Team B","Team B pick")
                decider()
            return played, log

        played, log = simulate_veto(team_a, team_b, fmt)

        st.subheader("🗺️ Map veto")
        col_v1, col_v2 = st.columns([1,1])
        with col_v1:
            log_df = pd.DataFrame(log)
            log_df.index = range(1, len(log_df)+1)
            st.dataframe(log_df, use_container_width=True)
        with col_v2:
            st.markdown("**Maps to be played:**")
            for i, v in enumerate(played):
                icons = {"Team A pick":"🔵","Team B pick":"🔴","Decider":"🟡"}
                st.markdown(f"**Map {i+1}:** {icons.get(v['type'],'⚪')} "
                            f"{v['map']} — *{v['type']}*")

        st.divider()

        # ── Step 3: FastAPI prediction ────────────────────────────────────────
        max_wins = 3 if fmt=="BO5" else 2 if fmt=="BO3" else 1
        score_a, score_b = 0, 0
        map_results = []

        with st.spinner("🔮 Running XGBoost prediction via FastAPI..."):
            for v in played:
                if score_a >= max_wins or score_b >= max_wins: break
                try:
                    resp = requests.post(f"{API_URL}/predict",
                                         json={"team_a":team_a,"team_b":team_b,
                                               "map_name":v["map"]}, timeout=15)
                    if resp.status_code == 200:
                        r = resp.json()
                        map_results.append({**v, **r})
                        if r["predicted_winner"] == team_a: score_a += 1
                        else: score_b += 1
                    else:
                        st.error(f"API error: {resp.text}")
                        st.stop()
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    st.stop()

        winner = team_a if score_a >= max_wins else team_b

        # Series result
        st.subheader("🏆 Series result")
        c1, c2, c3 = st.columns([2,1,2])
        with c1:
            if winner == team_a:
                st.success(f"🏆 **{team_a}**")
            else:
                st.error(f"**{team_a}**")
            st.metric("Maps won", score_a)
        with c2:
            st.markdown(
                f"<h2 style='text-align:center;margin-top:20px'>"
                f"{score_a} – {score_b}</h2>",
                unsafe_allow_html=True)
        with c3:
            if winner == team_b:
                st.success(f"🏆 **{team_b}**")
            else:
                st.error(f"**{team_b}**")
            st.metric("Maps won", score_b)

        st.info(f"🏆 **{winner}** predicted to win the {fmt} "
                f"series **{score_a}–{score_b}**")

        # Confidence meter
        st.subheader("📊 Confidence meter")
        conf_counts = {"High":0,"Medium":0,"Low":0}
        for r in map_results:
            c = r.get("confidence","Low")
            conf_counts[c] = conf_counts.get(c,0) + 1

        fig3, ax3 = plt.subplots(figsize=(6,0.8))
        left = 0
        for label, color in [("High","#00cc66"),("Medium","#ffaa00"),("Low","#ff4444")]:
            val = conf_counts[label]
            if val > 0:
                ax3.barh([""], [val], left=[left], color=color, label=f"{label} ({val})")
                left += val
        ax3.set_xlim(0, len(map_results) if map_results else 1)
        ax3.legend(loc="upper right", fontsize=9)
        ax3.set_title("Prediction confidence per map")
        ax3.set_yticks([])
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # Per map breakdown
        st.subheader("🗺️ Per-map breakdown")
        for i, r in enumerate(map_results):
            prob_a = r.get("team_a_win_prob", 0.5)
            prob_b = r.get("team_b_win_prob", 0.5)
            conf   = r.get("confidence","–")
            conf_icon = {"High":"🟢","Medium":"🟡","Low":"🔴"}.get(conf,"⚪")

            with st.expander(
                f"Map {i+1} — {r['map']} ({r['type']}) "
                f"→ {r.get('predicted_winner','?')} wins {conf_icon}"):
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(f"{team_a} win prob", f"{prob_a*100:.1f}%",
                              delta=f"{(prob_a-0.5)*100:+.1f}%")
                with m2:
                    st.metric("Confidence", conf)
                with m3:
                    st.metric(f"{team_b} win prob", f"{prob_b*100:.1f}%",
                              delta=f"{(prob_b-0.5)*100:+.1f}%")

                fig4, ax4 = plt.subplots(figsize=(6, 0.6))
                ax4.barh([""], [prob_a], color="#185FA5", label=team_a)
                ax4.barh([""], [prob_b], left=[prob_a], color="#993C1D", label=team_b)
                ax4.set_xlim(0,1)
                ax4.axvline(0.5, color='white', linewidth=1.5)
                ax4.legend(loc="upper right", fontsize=8)
                ax4.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax4.set_xticklabels(["0%","25%","50%","75%","100%"], fontsize=8)
                fig4.tight_layout()
                st.pyplot(fig4)
                plt.close()

        if n8n_triggered:
            st.success("📱 Telegram notification sent via n8n workflow!")

        with st.expander("📋 Raw API response (for report)"):
            st.json(map_results)
