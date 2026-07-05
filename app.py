from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd, numpy as np, json
from xgboost import XGBClassifier

app = FastAPI(title="VCT Match Predictor", version="1.0")

model = XGBClassifier()
model.load_model("model_artifacts/xgb_vct_model.json")
with open("model_artifacts/feature_cols.json") as f:
    FEATURE_COLS = json.load(f)
map_wr_df     = pd.read_csv("model_artifacts/map_wr_lookup.csv")
team_stats_df = pd.read_csv("model_artifacts/team_avg_stats.csv")

ROLE_MAP = {
    "Jett":"Duelist","Neon":"Duelist","Reyna":"Duelist",
    "Sova":"Initiator","Fade":"Initiator","Breach":"Initiator",
    "Astra":"Controller","Viper":"Controller","Brimstone":"Controller",
    "Killjoy":"Sentinel","Cypher":"Sentinel","Chamber":"Sentinel",
}
MAP_ORDER = ["Ascent","Bind","Breeze","Fracture","Haven",
             "Icebox","Lotus","Pearl","Split"]

def get_team(team):
    row = team_stats_df[team_stats_df["team"]==team]
    if row.empty: raise HTTPException(404, f"Team not found: {team}")
    return row.iloc[0].to_dict()

def get_map_wr(team, map_name):
    r = map_wr_df[(map_wr_df["team"]==team)&(map_wr_df["map"]==map_name)]
    return float(r["win_rate"].values[0]) if not r.empty else 0.5

class PredictReq(BaseModel):
    team_a: str; team_b: str; map_name: str
    agents_a: list[str]=[]; agents_b: list[str]=[]

class SeriesReq(BaseModel):
    team_a: str; team_b: str; format: str="bo3"

@app.get("/")
def root(): return {"status":"ok"}

@app.post("/predict")
def predict(req: PredictReq):
    sa, sb = get_team(req.team_a), get_team(req.team_b)
    row = {}
    for col in FEATURE_COLS:
        if col == "diff_map_wr":
            row[col] = get_map_wr(req.team_a,req.map_name) - get_map_wr(req.team_b,req.map_name)
        elif col == "map_enc":
            row[col] = MAP_ORDER.index(req.map_name) if req.map_name in MAP_ORDER else -1
        elif col.startswith("diff_role_balance"):
            ra = len({ROLE_MAP[a] for a in req.agents_a if a in ROLE_MAP})
            rb = len({ROLE_MAP[a] for a in req.agents_b if a in ROLE_MAP})
            row[col] = ra - rb
        else:
            stat = col.replace("diff_","")
            row[col] = sa.get(stat,0) - sb.get(stat,0)
    prob_a = float(model.predict_proba(pd.DataFrame([row])[FEATURE_COLS])[0][1])
    conf   = "High" if abs(prob_a-.5)>.25 else ("Medium" if abs(prob_a-.5)>.12 else "Low")
    return {"team_a":req.team_a,"team_b":req.team_b,"map":req.map_name,
            "team_a_win_prob":round(prob_a,4),"team_b_win_prob":round(1-prob_a,4),
            "predicted_winner":req.team_a if prob_a>=.5 else req.team_b,"confidence":conf}

@app.post("/series")
def series(req: SeriesReq):
    pool=["Ascent","Pearl","Haven","Lotus","Fracture","Bind","Split"]
    used,played,log=[],[],[]
    def wr(t,m): return get_map_wr(t,m)
    def ban(t,n):
        w=sorted([m for m in pool if m not in used],key=lambda m:wr(t,m))[0]
        used.append(w);log.append({"action":"ban","team":n,"map":w})
    def pick(t,n,tag):
        b=sorted([m for m in pool if m not in used],key=lambda m:wr(t,m),reverse=True)[0]
        used.append(b);played.append({"map":b,"type":tag})
        log.append({"action":"pick","team":n,"map":b})
    def decider():
        r=[m for m in pool if m not in used][0]
        used.append(r);played.append({"map":r,"type":"decider"})
        log.append({"action":"decider","team":"–","map":r})
    fmt=req.format.lower()
    if fmt=="bo1":
        for _ in range(3): ban(req.team_a,req.team_a);ban(req.team_b,req.team_b)
        decider()
    elif fmt=="bo3":
        ban(req.team_a,req.team_a);ban(req.team_b,req.team_b)
        ban(req.team_a,req.team_a);ban(req.team_b,req.team_b)
        pick(req.team_a,req.team_a,"pick_a");pick(req.team_b,req.team_b,"pick_b");decider()
    else:
        ban(req.team_a,req.team_a);ban(req.team_b,req.team_b)
        pick(req.team_a,req.team_a,"pick_a");pick(req.team_a,req.team_a,"pick_a")
        pick(req.team_b,req.team_b,"pick_b");pick(req.team_b,req.team_b,"pick_b");decider()
    mx=3 if fmt=="bo5" else 2 if fmt=="bo3" else 1
    sa,sb,results=0,0,[]
    for v in played:
        if sa>=mx or sb>=mx: break
        p=predict(PredictReq(team_a=req.team_a,team_b=req.team_b,map_name=v["map"]))
        if p["predicted_winner"]==req.team_a: sa+=1
        else: sb+=1
        results.append({**v,**p})
    return {"format":fmt.upper(),"team_a":req.team_a,"team_b":req.team_b,
            "score":f"{sa}–{sb}",
            "series_winner":req.team_a if sa>=mx else req.team_b,
            "veto_log":log,"map_results":results}