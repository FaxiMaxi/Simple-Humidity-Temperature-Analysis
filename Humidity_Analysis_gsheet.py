import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import warnings
from streamlit_gsheets import GSheetsConnection

# -----------------------------------------------------------------------------
# 1. Configuration & Global Settings
# -----------------------------------------------------------------------------
# Suppress pandas warnings about future behavior
warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation")

st.set_page_config(
    page_title="Sensor Analytics Cloud",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Ensure sidebar toggle remains visible */
    div.block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Logic: Data Processing & Cleaning
# -----------------------------------------------------------------------------
def normalize_columns(df):
    """Ensures consistent column names."""
    df = df.copy()
    expected_cols = ['Time', 'Sensor ID', 'Temperature', 'Humidity']
    # If columns don't match names but count is right, rename them
    if len(df.columns) >= 4:
        # Create a map only for the first 4 columns
        rename_map = {df.columns[i]: expected_cols[i] for i in range(4)}
        df = df.rename(columns=rename_map)
    return df

def process_data(df_input):
    df = df_input.copy()
    
    # --- A. Time Column Cleanup ---
    if 'Time' in df.columns:
        # 1. Convert to string & strip spaces
        df['Time'] = df['Time'].astype(str).str.strip()
        # 2. Replace empty/nan strings with actual NaN
        df['Time'] = df['Time'].replace(['nan', 'None', '', 'NaT'], np.nan)
        # 3. Forward Fill (Gap Filling)
        df['Time'] = df['Time'].ffill()
    
    # --- B. Datetime Conversion (Robust) ---
    # Strategy: Try fast format first, fall back to flexible parser
    df['Datetime'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce')
    
    # Fallback for mixed inputs (e.g. 10:00:05 or 10:00 AM)
    mask_nat = df['Datetime'].isna() & df['Time'].notna()
    if mask_nat.any():
        df.loc[mask_nat, 'Datetime'] = pd.to_datetime(df.loc[mask_nat, 'Time'], errors='coerce')
    
    # --- C. Numeric Conversion (Comma Support) ---
    cols_to_numeric = ['Sensor ID', 'Temperature', 'Humidity']
    for col in cols_to_numeric:
        if col in df.columns:
            # Replace comma with dot for EU format (e.g. "10,5" -> "10.5")
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# -----------------------------------------------------------------------------
# 3. Data Loading (Google Sheets)
# -----------------------------------------------------------------------------
st.sidebar.header("📂 Cloud Data")

# Initialize Connection
# This looks for [connections.gsheets] in .streamlit/secrets.toml
conn = st.connection("gsheets", type=GSheetsConnection)

@st.cache_data(ttl=5) # Cache data for 5 seconds to prevent spamming API
def load_data():
    try:
        # Read from the default spreadsheet URL in secrets
        df = conn.read(worksheet="Sheet1")
        return df
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return pd.DataFrame()

# Button to manually refresh
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Load the data
df_raw = load_data()

# Helper: Create empty structure if sheet is totally blank
if df_raw.empty:
    df_raw = pd.DataFrame({
        'Time': ['10:00', '10:05'], 
        'Sensor ID': [1, 2], 
        'Temperature': [20.5, 21.0], 
        'Humidity': [50, 55]
    })

# Normalize headers (in case Sheet headers are wrong)
df_raw = normalize_columns(df_raw)

# -----------------------------------------------------------------------------
# 4. Main Interface
# -----------------------------------------------------------------------------
st.title("📊 Sensor Analytics (Cloud)")

# --- EDITOR & SAVE ---
col_edit, col_save = st.columns([4, 1])
with col_edit:
    st.subheader("📝 Live Data Editor")
    st.caption("Changes are saved directly to Google Sheets.")
with col_save:
    save_clicked = st.button("💾 Save to Cloud", type="primary", use_container_width=True)

# Data Editor
edited_df = st.data_editor(
    df_raw,
    num_rows="dynamic",
    width="stretch",
    key="editor"
)

# SAVE LOGIC
if save_clicked:
    try:
        with st.spinner("Saving to Google Sheets..."):
            # Update the Google Sheet with current editor data
            conn.update(worksheet="Sheet1", data=edited_df)
            st.success("✅ Saved successfully!")
            st.cache_data.clear() # Clear cache so next load gets new data
    except Exception as e:
        st.error(f"❌ Save failed: {e}")

# -----------------------------------------------------------------------------
# 5. Analytics & Visualization
# -----------------------------------------------------------------------------
# Process data for plotting
try:
    df = process_data(edited_df)
    # Filter valid rows for plotting
    plot_df = df.dropna(subset=['Datetime', 'Sensor ID']).sort_values(by=['Datetime', 'Sensor ID'])
except Exception as e:
    st.error(f"Processing Error: {e}")
    plot_df = pd.DataFrame()

if plot_df.empty:
    st.warning("No valid data to visualize.")
    st.stop()

# --- PLOTS ---
st.divider()
st.subheader("📈 Analytics")

def configure_chart(fig):
    fig.update_xaxes(tickformat="%H:%M", showgrid=True)
    return fig

# A. Averages
avg_df = plot_df.groupby('Datetime')[['Temperature', 'Humidity']].mean().reset_index()
c1, c2 = st.columns(2)

fig_t = px.area(avg_df, x='Datetime', y='Temperature', title="Avg Temp (°C)", 
                color_discrete_sequence=['#FF4B4B'], markers=True)
c1.plotly_chart(configure_chart(fig_t), width="stretch", key="avg_temp")

fig_h = px.area(avg_df, x='Datetime', y='Humidity', title="Avg Humidity (%)", 
                color_discrete_sequence=['#0068C9'], markers=True)
c2.plotly_chart(configure_chart(fig_h), width="stretch", key="avg_hum")

# B. Individual Sensors
st.divider()
sensors = sorted(plot_df['Sensor ID'].unique())

if sensors:
    tabs = st.tabs([f"Sensor {int(s)}" for s in sensors])
    for tab, sid in zip(tabs, sensors):
        with tab:
            s_data = plot_df[plot_df['Sensor ID'] == sid]
            col_a, col_b = st.columns(2)
            
            # Temp
            ft = px.line(s_data, x='Datetime', y='Temperature', title="Temperature", 
                         markers=True, color_discrete_sequence=['#FF4B4B'])
            col_a.plotly_chart(configure_chart(ft), width="stretch", key=f"temp_{sid}")
            
            # Humidity
            fh = px.line(s_data, x='Datetime', y='Humidity', title="Humidity", 
                         markers=True, color_discrete_sequence=['#0068C9'])
            col_b.plotly_chart(configure_chart(fh), width="stretch", key=f"hum_{sid}")
