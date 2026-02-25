import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
from scipy.stats import poisson
import scipy.stats as stats
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="HSSE Reliability Engine", layout="wide")

# --- 1. AUTHENTICATION ---
try:
    VALID_USERS = st.secrets["passwords"]
except Exception:
    VALID_USERS = {"admin": "Safety2026", "site_manager": "CharlieCheck123", "external_auditor": "ExternalPass!7"}

def login_screen():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        st.title("🔒 HSSE Reliability Engine")
        user = st.text_input("Username").lower()
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if user in VALID_USERS and VALID_USERS[user] == password:
                st.session_state["authenticated"], st.session_state["current_user"] = True, user
                st.rerun()
            else:
                st.error("Invalid credentials")
        return False
    return True

if login_screen():
    st.sidebar.write(f"👤 User: **{st.session_state['current_user']}**")
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"authenticated": False}))
    
    # --- 2. DATA LOADING ---
    uploaded_file = st.sidebar.file_uploader("Upload HSSE CSV", type="csv")
    if not uploaded_file:
        st.info("Please upload your HSSE CSV to activate the engine.")
        st.stop()

    df = pd.read_csv(uploaded_file)
    df['log_exposure'] = np.log(df['Man_Hours'])
    df['Incident_Rate'] = (df['Incidents'] / df['Man_Hours']) * 200000

    # --- 3. BENCHMARK & FILTERS ---
    st.sidebar.markdown("---")
    industry_standards = {"Construction": 2.5, "Manufacturing": 1.8, "Oil & Gas": 0.8, "Custom Target": 1.0}
    benchmark_type = st.sidebar.selectbox("Industry Benchmark:", list(industry_standards.keys()))
    benchmark_val = industry_standards[benchmark_type]

    selected_site = st.sidebar.multiselect("Filter Site", options=df['Site'].unique(), default=df['Site'].unique())
    f_df = df[df['Site'].isin(selected_site)]

    # --- 4. PREDICTIVE MODEL TRAINING ---
    model = smf.glm(formula="Incidents ~ Audits + Observations + Maint_Compliance", 
                    data=f_df, family=sm.families.Poisson(), offset=f_df['log_exposure']).fit()

    # --- 5. SIMULATOR & SCENARIO STORAGE (MOVED UP) ---
    st.title("🚀 HSSE Strategy & Prediction Simulator")
    
    HISTORY_FILE = "scenario_history.json"
    def save_set(user, name, a, o, m):
        hist = json.load(open(HISTORY_FILE)) if os.path.exists(HISTORY_FILE) else {}
        if user not in hist: hist[user] = {}
        hist[user][name] = {"A": a, "O": o, "M": m}
        json.dump(hist, open(HISTORY_FILE, "w"))

    user_hist = (json.load(open(HISTORY_FILE)) if os.path.exists(HISTORY_FILE) else {}).get(st.session_state["current_user"], {})
    
    col_s1, col_s2 = st.columns([2, 1])
    with col_s2:
        load_name = st.selectbox("📂 Load Scenario", ["Manual Entry"] + list(user_hist.keys()))
    
    d_a, d_o, d_m = (user_hist[load_name]["A"], user_hist[load_name]["O"], user_hist[load_name]["M"]) if load_name != "Manual Entry" else (int(f_df['Audits'].mean()), int(f_df['Observations'].mean()), int(f_df['Maint_Compliance'].mean()))
    
    with col_s1:
        s1, s2, s3 = st.columns(3)
        aud = s1.slider("Audits", 0, 50, d_a)
        obs = s2.slider("Observations", 0, 100, d_o)
        mnt = s3.slider("Maint %", 0, 100, d_m)
    
    # --- 6. KPIs & PERCENTILE RANKING ---
    st.markdown("---")
    avg_rate = (f_df['Incidents'].sum() / f_df['Man_Hours'].sum() * 200000) if f_df['Man_Hours'].sum() > 0 else 0
    percentile = (1 - stats.lognorm.cdf(avg_rate, s=0.6, scale=benchmark_val)) * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("Actual Incident Rate", f"{avg_rate:.2f}", delta=f"{avg_rate-benchmark_val:.2f} vs Bench", delta_color="inverse")
    k2.metric("Industry Standing", f"{percentile:.1f}%")
    
    # Calculate Predicted Rate based on Slider inputs
    pred_data = pd.DataFrame({'Audits':[aud],'Observations':[obs],'Maint_Compliance':[mnt]})
    pred_count = model.predict(pred_data, offset=[np.log(f_df['Man_Hours'].mean())])[0]
    pred_rate_val = (pred_count / f_df['Man_Hours'].mean()) * 200000
    k3.metric("Predicted Rate (Simulated)", f"{pred_rate_val:.2f}", delta=f"{pred_rate_val-avg_rate:.2f} vs Actual", delta_color="inverse")

    # --- 7. DYNAMIC HEATMAPS ---
    st.markdown("---")
    st.subheader("🔥 Risk Map: Current Reality vs. Simulated Future")
    
    actual_pivot = f_df.pivot_table(index='Contractor', columns='Site', values='Incident_Rate', aggfunc='mean').fillna(0)
    
    # Build grid based on SLIDER values
    grid_rows = []
    for s in f_df['Site'].unique():
        for c in f_df['Contractor'].unique():
            grid_rows.append({'Site': s, 'Contractor': c, 'Audits': aud, 'Observations': obs, 'Maint_Compliance': mnt, 'Man_Hours': f_df['Man_Hours'].mean()})
    
    grid_df = pd.DataFrame(grid_rows)
    grid_df['log_exposure'] = np.log(grid_df['Man_Hours'])
    grid_df['Predicted_Rate'] = (model.predict(grid_df, offset=grid_df['log_exposure']) / grid_df['Man_Hours']) * 200000
    pred_pivot = grid_df.pivot_table(index='Contractor', columns='Site', values='Predicted_Rate').fillna(0)

    max_v = max(actual_pivot.max().max(), pred_pivot.max().max())
    c_h1, c_h2 = st.columns(2)
    c_h1.plotly_chart(px.imshow(actual_pivot, text_auto=".2f", color_continuous_scale='Reds', range_color=[0, max_v], title="Actual Historical Rates"), use_container_width=True)
    c_h2.plotly_chart(px.imshow(pred_pivot, text_auto=".2f", color_continuous_scale='Reds', range_color=[0, max_v], title="Simulated Risk Level"), use_container_width=True)

    # --- 8. SAVE FEATURE ---
    st.sidebar.markdown("---")
    scen_name = st.sidebar.text_input("Scenario Name", placeholder="e.g. 2026 Goal")
    if st.sidebar.button("💾 Save Current Simulation"):
        if scen_name:
            save_set(st.session_state["current_user"], scen_name, aud, obs, mnt)
            st.sidebar.success(f"Saved {scen_name}")
            st.rerun()