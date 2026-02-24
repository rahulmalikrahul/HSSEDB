import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly.express as px
from scipy.stats import poisson

# 1. Define your users (In a real app, use Streamlit Secrets for this)
VALID_USERS = {
    "admin": "Safety2026",
    "site_manager": "CharlieCheck123",
    "external_auditor": "ExternalPass!7"
}

def login_screen():
    """Returns True if the user is authenticated."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔒 HSSE Engine Access")
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if user in VALID_USERS and VALID_USERS[user] == password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid username or password")
        return False
    return True

# --- MAIN APP LOGIC ---
if login_screen():
    # Put all your existing app code (KPIs, Heatmap, Model) inside this block
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"authenticated": False}))
    
    # YOUR EXISTING CODE STARTS HERE...
    #st.title("🛡️ HSSE Reliability & Drill-Down Dashboard")
    # ... (rest of your app)


# --- PAGE CONFIG ---
st.set_page_config(page_title="HSSE Reliability Engine", layout="wide")

st.title("🛡️ HSSE Reliability & Drill-Down Dashboard")

# --- 1. DATA LOADING ---
uploaded_file = st.sidebar.file_uploader("Upload HSSE CSV", type="csv")

if not uploaded_file:
    st.info("Please upload your HSSE CSV to activate the Dashboard.")
    st.stop()

df = pd.read_csv(uploaded_file)
df['log_exposure'] = np.log(df['Man_Hours'])
df['Incident_Rate'] = (df['Incidents'] / df['Man_Hours']) * 200000

# --- 2. GLOBAL DRILL-DOWN FILTERS ---
st.sidebar.header("🔍 Drill-Down Filters")
selected_site = st.sidebar.multiselect("Filter by Site", options=df['Site'].unique(), default=df['Site'].unique())
selected_contractor = st.sidebar.multiselect("Filter by Contractor", options=df['Contractor'].unique(), default=df['Contractor'].unique())

# Filtered Data for Drill-Down
f_df = df[(df['Site'].isin(selected_site)) & (df['Contractor'].isin(selected_contractor))]

# --- 3. EXECUTIVE SUMMARY (Top Row) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Incidents", f_df['Incidents'].sum())
col2.metric("Total Exposure (Hrs)", f"{f_df['Man_Hours'].sum():,}")
col3.metric("Avg Incident Rate", f"{(f_df['Incidents'].sum()/f_df['Man_Hours'].sum()*200000):.2f}")
col4.metric("Risk Rating", "High" if (f_df['Incidents'].sum()/f_df['Man_Hours'].sum()*200000) > 1.0 else "Stable")

# --- 4. THE HEATMAP (Interaction Risk) ---
st.markdown("---")
st.subheader("🔥 Site-Contractor Interaction Heatmap")

# Pivot data for heatmap
heat_map_data = f_df.pivot_table(index='Contractor', columns='Site', values='Incident_Rate', aggfunc='mean').fillna(0)
fig_heat = px.imshow(heat_map_data, text_auto=".2f", color_continuous_scale='Reds', labels=dict(color="Incident Rate"))
st.plotly_chart(fig_heat, use_container_width=True)

# --- 5. STATISTICAL RELIABILITY DRILL-DOWN ---
st.markdown("---")
tab1, tab2 = st.tabs(["🚀 Predictive Simulator", "📊 Leading Indicator Analysis"])

with tab1:
    st.subheader("What-If Simulation (Based on Current Selection)")
    # Model built only on filtered data
    try:
        model = smf.glm(formula="Incidents ~ Audits + Observations + Maint_Compliance", 
                        data=f_df, family=sm.families.Poisson(), offset=f_df['log_exposure']).fit()
        
        # Sliders for simulation
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1: aud = st.slider("Target Audits", 0, 50, int(f_df['Audits'].mean()))
        with s_col2: obs = st.slider("Target Observations", 0, 100, int(f_df['Observations'].mean()))
        with s_col3: maint = st.slider("Maint. Compliance %", 0, 100, int(f_df['Maint_Compliance'].mean()))
        
        # Calculate Prediction
        pred_data = pd.DataFrame({'Audits': [aud], 'Observations': [obs], 'Maint_Compliance': [maint]})
        lambda_val = model.predict(pred_data, offset=[np.log(f_df['Man_Hours'].mean())])[0]
        
        st.write(f"### Predicted Probability: { (1 - poisson.pmf(0, lambda_val))*100:.1f}% chance of incident.")
        
    except:
        st.warning("Not enough data in current filter to run the statistical model. Broaden your selection.")

with tab2:
    st.subheader("Leading vs Lagging Correlation")
    fig_scatter = px.scatter(f_df, x="Observations", y="Incidents", size="Man_Hours", 
                             color="Contractor", hover_name="Site", trendline="ols")
    st.plotly_chart(fig_scatter, use_container_width=True)