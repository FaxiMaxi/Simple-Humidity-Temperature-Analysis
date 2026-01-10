import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import warnings
from streamlit_gsheets import GSheetsConnection

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of DataFrame concatenation",
)

st.set_page_config(
    page_title="Sensor Analytics Cloud",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  div.block-container {padding-top: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Google Sheets connection
# -----------------------------------------------------------------------------
conn = st.connection("gsheets", type=GSheetsConnection)
SPREADSHEET_URL = st.secrets["connections"]["gsheets"]["spreadsheet"]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
CANON_VIEW_COLS = ["Time", "Sensor ID", "Temperature", "Humidity"]
CANON_SHEET_COLS = ["Row"] + CANON_VIEW_COLS


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = list(df.columns)

    if len(cols) >= 5:
        rename_map = {
            cols[0]: "Row",
            cols[1]: "Time",
            cols[2]: "Sensor ID",
            cols[3]: "Temperature",
            cols[4]: "Humidity",
        }
        df = df.rename(columns=rename_map)
    elif len(cols) >= 4:
        rename_map = {
            cols[0]: "Time",
            cols[1]: "Sensor ID",
            cols[2]: "Temperature",
            cols[3]: "Humidity",
        }
        df = df.rename(columns=rename_map)

    return df


def ensure_row_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Row" not in df.columns:
        df.insert(0, "Row", range(1, len(df) + 1))
        return df

    df["Row"] = pd.to_numeric(df["Row"], errors="coerce")
    max_existing = int(df["Row"].dropna().max()) if df["Row"].notna().any() else 0

    missing = df["Row"].isna()
    if missing.any():
        new_ids = range(max_existing + 1, max_existing + 1 + int(missing.sum()))
        df.loc[missing, "Row"] = list(new_ids)

    df["Row"] = df["Row"].astype(int)
    df = df.sort_values("Row", kind="stable").reset_index(drop=True)
    return df


def process_data(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()

    if "Time" in df.columns:
        df["Time"] = df["Time"].astype(str).str.strip()
        df["Time"] = df["Time"].replace(["nan", "None", "", "NaT"], np.nan)
        df["Time"] = df["Time"].ffill()

    # Fast HH:MM, then fallback
    df["Datetime"] = pd.to_datetime(df.get("Time", pd.Series(dtype="object")), format="%H:%M", errors="coerce")
    mask_nat = df["Datetime"].isna() & df.get("Time", pd.Series(dtype="object")).notna()
    if mask_nat.any():
        df.loc[mask_nat, "Datetime"] = pd.to_datetime(df.loc[mask_nat, "Time"], errors="coerce")

    # Numeric + decimal comma
    for col in ["Sensor ID", "Temperature", "Humidity"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_worksheet_titles() -> list[str]:
    client = conn.client
    ss = client._open_spreadsheet(spreadsheet=SPREADSHEET_URL)
    return [ws.title for ws in ss.worksheets()]


def make_default_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Row": [1, 2, 3, 4],
            "Time": ["10:00", "", "", ""],
            "Sensor ID": [1, 2, 3, 4],
            "Temperature": [20.5, 21.0, 19.5, 20.2],
            "Humidity": [45, 46, 50, 44],
        }
    )


def apply_line_style(fig, line_color: str, smoothing: float):
    """
    Makes all traces in a figure use spline + smoothing + chosen color.
    Smoothing is effective with spline lines. [web:91][web:104]
    """
    fig.update_traces(line=dict(shape="spline", smoothing=float(smoothing), color=line_color))
    return fig


def configure_time_axis(fig):
    fig.update_xaxes(tickformat="%H:%M", showgrid=True)
    return fig


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Google Sheet")

try:
    worksheet_titles = get_worksheet_titles()
except Exception as e:
    st.sidebar.error(f"Could not list worksheet tabs: {e}")
    worksheet_titles = ["Sheet1"]

selected_ws = st.sidebar.radio(
    "Worksheet tab",
    options=worksheet_titles,
    index=worksheet_titles.index("Sheet1") if "Sheet1" in worksheet_titles else 0,
)

if st.sidebar.button("🔄 Refresh"):
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Chart styling")

smoothness = st.sidebar.slider(
    "Line smoothness",
    min_value=0.0,
    max_value=1.3,
    value=0.9,
    step=0.1,
    help="Plotly spline smoothing (0 = almost linear, higher = smoother).",
)

avg_temp_color = st.sidebar.color_picker("Avg temperature color", "#FF4B4B")  # [web:96]
avg_hum_color = st.sidebar.color_picker("Avg humidity color", "#0068C9")      # [web:96]
sensor_temp_color = st.sidebar.color_picker("Sensor temperature color", "#FF4B4B")  # [web:96]
sensor_hum_color = st.sidebar.color_picker("Sensor humidity color", "#0068C9")      # [web:96]

# -----------------------------------------------------------------------------
# Load sheet
# -----------------------------------------------------------------------------
df_raw = conn.read(worksheet=selected_ws, ttl=5)

if df_raw is None or df_raw.empty:
    df_raw = make_default_df()

df_raw = ensure_row_column(normalize_columns(df_raw))

# INTERNAL (has Row), VIEW (no Row shown)
df_view = df_raw[CANON_VIEW_COLS].copy()

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("📊 Sensor Analytics (Cloud)")
st.caption(f"Active worksheet: {selected_ws}")

c1, c2 = st.columns([4, 1])
with c1:
    st.subheader("📝 Data Editor")
    st.caption("Row IDs are managed automatically (hidden). Empty Time cells are forward-filled for analysis.")
with c2:
    save_clicked = st.button("💾 Save", type="primary")

edited_view = st.data_editor(
    df_view,
    num_rows="dynamic",
    width="stretch",
    key=f"editor_{selected_ws}",
)  # dynamic rows supported here [web:21]

# Save: re-add Row before writing to the sheet
if save_clicked:
    try:
        to_save = edited_view.copy()
        to_save.insert(0, "Row", range(1, len(to_save) + 1))
        to_save = to_save[CANON_SHEET_COLS]
        conn.update(worksheet=selected_ws, data=to_save)
        st.success("Saved to Google Sheets.")
    except Exception as e:
        st.error(f"Save failed: {e}")

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
df_for_processing = edited_view.copy()
df_for_processing.insert(0, "Row", range(1, len(df_for_processing) + 1))

df = process_data(df_for_processing)
plot_df = df.dropna(subset=["Datetime", "Sensor ID"]).sort_values(["Datetime", "Sensor ID"])

if plot_df.empty:
    st.warning("No valid data to visualize (check Time + Sensor ID columns).")
    st.stop()

st.divider()
st.subheader("📈 Network averages")

avg_df = plot_df.groupby("Datetime")[["Temperature", "Humidity"]].mean().reset_index()
a, b = st.columns(2)

fig_avg_temp = px.area(avg_df, x="Datetime", y="Temperature", title="Avg Temperature (°C)", markers=True)
fig_avg_temp = configure_time_axis(fig_avg_temp)
fig_avg_temp = apply_line_style(fig_avg_temp, avg_temp_color, smoothness)
a.plotly_chart(fig_avg_temp, width="stretch", key=f"avg_temp_{selected_ws}")

fig_avg_hum = px.area(avg_df, x="Datetime", y="Humidity", title="Avg Humidity (%)", markers=True)
fig_avg_hum = configure_time_axis(fig_avg_hum)
fig_avg_hum = apply_line_style(fig_avg_hum, avg_hum_color, smoothness)
b.plotly_chart(fig_avg_hum, width="stretch", key=f"avg_hum_{selected_ws}")

st.divider()
st.subheader("🔍 Individual sensors")

sensors = sorted([s for s in plot_df["Sensor ID"].unique() if pd.notna(s)])
tabs = st.tabs([f"Sensor {int(s)}" for s in sensors])

for tab, sid in zip(tabs, sensors):
    with tab:
        s_data = plot_df[plot_df["Sensor ID"] == sid]
        left, right = st.columns(2)

        fig_t = px.line(s_data, x="Datetime", y="Temperature", title="Temperature", markers=True)
        fig_t = configure_time_axis(fig_t)
        fig_t = apply_line_style(fig_t, sensor_temp_color, smoothness)
        left.plotly_chart(fig_t, width="stretch", key=f"temp_{selected_ws}_{sid}")

        fig_h = px.line(s_data, x="Datetime", y="Humidity", title="Humidity", markers=True)
        fig_h = configure_time_axis(fig_h)
        fig_h = apply_line_style(fig_h, sensor_hum_color, smoothness)
        right.plotly_chart(fig_h, width="stretch", key=f"hum_{selected_ws}_{sid}")
