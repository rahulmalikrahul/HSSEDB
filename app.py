import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
from scipy.stats import poisson, lognorm
import scipy.stats as stats
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="HSSE Reliability Engine", layout="wide")

# --- 1. AUTHENTICATION LOGIC ---
try:
    VALID_USERS = st.secrets["passwords"]
except Exception:
    # Local fallback for testing
    VALID_USERS = {
        "admin": "Safety2026", 
        "site_manager": "CharlieCheck123", 
        "external_auditor": "ExternalPass!7"
    }

def login_screen():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔒 HSSE Reliability Engine")
        st.info("Please log in with your assigned credentials.")
        user = st.text_input("Username").lower()
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if user in VALID_USERS and VALID_USERS[user] == password:
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = user
                st.rerun()
            else:
                st.error("Invalid username or password")
        return False
    return True

if login_screen():
    # Sidebar Setup
    st.sidebar.write(f"👤 Logged in as: **{st.session_state['current_user']}**")
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"authenticated": False}))
    
    # --- 2. DATA LOADING ---
    uploaded_file = st.sidebar.file_uploader("Upload HSSE CSV", type="csv")
    if not uploaded_file:
        st.info("Waiting for data. Please upload your HSSE CSV to begin.")
        st.stop()

    df = pd.read_csv(uploaded_file)
    df['log_exposure'] = np.log(df['Man_Hours'])
    df['Incident_Rate'] = (df['Incidents'] / df['Man_Hours']) * 200000

    # --- 3. BENCHMARKING & FILTERS ---
    st.sidebar.markdown("---")
    industry_standards = {
        "Construction": 2.5, "Manufacturing": 1.8, 
        "Oil & Gas": 0.8, "Custom Target": 1.0
    }
    benchmark_type = st.sidebar.selectbox("Industry Benchmark:", list(industry_standards.keys()))
    benchmark_val = industry_standards[benchmark_type]

    selected_site = st.sidebar.multiselect("Filter Site", options=df['Site'].unique(), default=df['Site'].unique())
    f_df = df[df['Site'].isin(selected_site)]

    # --- 4. PERCENTILE RANKING & KPIs ---
    avg_rate = (f_df['Incidents'].sum() / f_df['Man_Hours'].sum() * 200000) if f_df['Man_Hours'].sum() > 0 else 0
    
    # Percentile Math (Log-Normal Distribution)
    percentile = stats.lognorm.cdf(avg_rate, s=0.6, scale=benchmark_val)
    percentile_rank = (1 - percentile) * 100

    st.title("📊 Safety Performance & Industry Ranking")
    
    col_k1, col_k2, col_k3 = st.columns(3)
    col_k1.metric("Your Incident Rate", f"{avg_rate:.2f}", delta=f"{avg_rate-benchmark_val:.2f} vs Bench", delta_color="inverse")
    col_k2.metric("Industry Standing", f"{percentile_rank:.1f}%", help="Higher is better. 90% means you are in the top 10% of safety performers.")
    
    status = "Leader" if percentile_rank > 75 else "Average" if percentile_rank > 40 else "At Risk"
    col_k3.metric("Performance Status", status)

    # --- 5. MODEL TRAINING ---
    model = smf.glm(formula="Incidents ~ Audits + Observations + Maint_Compliance", 
                    data=f_df, family=sm.families.Poisson(), offset=f_df['log_exposure']).fit()

    # --- 6. HEATMAPS (ACTUAL vs PREDICTED) ---
    st.markdown("---")
    st.subheader("🔥 Risk Assessment: Actual vs. Predicted")
    
    with st.expander("🎓 WalkMe: How to read this"):
        st.info("The Right map shows where the model 'expects' trouble based on your current safety metrics. Compare it to the Left map to see if you have hidden risks.")

    actual_pivot = f_df.pivot_table(index='Contractor', columns='Site', values='Incident_Rate', aggfunc='mean').fillna(0)
    
    # Create Prediction Grid
    grid_rows = []
    for s in f_df['Site'].unique():
        for c in f_df['Contractor'].unique():
            grid_rows.append({'Site': s, 'Contractor': c, 'Audits': f_df['Audits'].mean(), 
                              'Observations': f_df['Observations'].mean(), 'Maint_Compliance': f_df['Maint_Compliance'].mean(),
                              'Man_Hours': f_df['Man_Hours'].mean()})
    grid_df = pd.DataFrame(grid_rows)
    grid_df['log_exposure'] = np.log(grid_df['Man_Hours'])
    grid_df['Predicted_Rate'] = (model.predict(grid_df, offset=grid_df['log_exposure']) / grid_df['Man_Hours']) * 200000
    pred_pivot = grid_df.pivot_table(index='Contractor', columns='Site', values='Predicted_Rate').fillna(0)

    max_v = max(actual_pivot.max().max(), pred_pivot.max().max())
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.imshow(actual_pivot, text_auto=".2f", color_continuous_scale='Reds', range_color=[0, max_v], title="Actual Rates"), use_container_width=True)
    c2.plotly_chart(px.imshow(pred_pivot, text_auto=".2f", color_continuous_scale='Reds', range_color=[0, max_v], title="Predicted Risk"), use_container_width=True)

    # --- 7. SCENARIO SAVING & SIMULATOR ---
    st.markdown("---")
    st.subheader("🚀 Predictive Simulator")
    
    HISTORY_FILE = "scenario_history.json"
    def save_set(user, name, a, o, m):
        hist = json.load(open(HISTORY_FILE)) if os.path.exists(HISTORY_FILE) else {}
        if user not in hist: hist[user] = {}
        hist[user][name] = {"A": a, "O": o, "M": m}
        json.dump(hist, open(HISTORY_FILE, "w"))

    user_hist = (json.load(open(HISTORY_FILE)) if os.path.exists(HISTORY_FILE) else {}).get(st.session_state["current_user"], {})
    load_name = st.selectbox("📂 Load Saved Scenario", ["New"] + list(user_hist.keys()))
    
    d_a, d_o, d_m = (user_hist[load_name]["A"], user_hist[load_name]["O"], user_hist[load_name]["M"]) if load_name != "New" else (10, 20, 85)
    
    s1, s2, s3 = st.columns(3)
    aud = s1.slider("Audits", 0, 50, d_a)
    obs = s2.slider("Observations", 0, 100, d_o)
    mnt = s3.slider("Maint %", 0, 100, d_m)
    
    if st.button("💾 Save Scenario"):
        save_set(st.session_state["current_user"], f"Scenario_{len(user_hist)+1}", aud, obs, mnt)
        st.success("Saved!")

    # Final Probability Result
    res = (1 - poisson.pmf(0, model.predict(pd.DataFrame({'Audits':[aud],'Observations':[obs],'Maint_Compliance':[mnt]}), 
                                            offset=[np.log(f_df['Man_Hours'].mean())])[0])) * 100
    st.metric("Probability of Incident", f"{res:.1f}%")