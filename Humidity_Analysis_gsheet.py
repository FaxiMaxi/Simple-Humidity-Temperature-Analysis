import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import warnings
from streamlit_gsheets import GSheetsConnection

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

# ---------- Google Sheets connection ----------
conn = st.connection("gsheets", type=GSheetsConnection)

# Expect this to exist in secrets.toml:
# [connections.gsheets]
# spreadsheet = "https://docs.google.com/spreadsheets/d/...."
SPREADSHEET = st.secrets["connections"]["gsheets"]["spreadsheet"]


# ---------- Helpers ----------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supports two layouts:
      A) Time, Sensor ID, Temperature, Humidity
      B) Row, Time, Sensor ID, Temperature, Humidity   (recommended)
    Renames by *position* to avoid header mismatch issues.
    """
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
        return df.rename(columns=rename_map)

    if len(cols) >= 4:
        rename_map = {
            cols[0]: "Time",
            cols[1]: "Sensor ID",
            cols[2]: "Temperature",
            cols[3]: "Humidity",
        }
        return df.rename(columns=rename_map)

    return df


def ensure_row_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures there is a 'Row' column and it is filled with integers.
    - Existing Row values are preserved.
    - Missing Row values are auto-assigned (max+1 ...).
    """
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

    # Keep Row numeric if present
    if "Row" in df.columns:
        df["Row"] = pd.to_numeric(df["Row"], errors="coerce")

    # --- Time cleanup + forward-fill ---
    if "Time" in df.columns:
        df["Time"] = df["Time"].astype(str).str.strip()
        df["Time"] = df["Time"].replace(["nan", "None", "", "NaT"], np.nan)
        df["Time"] = df["Time"].ffill()

    # --- Datetime conversion (fast path + fallback) ---
    df["Datetime"] = pd.to_datetime(df.get("Time", pd.Series(dtype="object")), format="%H:%M", errors="coerce")
    mask_nat = df["Datetime"].isna() & df.get("Time", pd.Series(dtype="object")).notna()
    if mask_nat.any():
        df.loc[mask_nat, "Datetime"] = pd.to_datetime(df.loc[mask_nat, "Time"], errors="coerce")

    # --- Numeric conversion + comma decimal support ---
    for col in ["Sensor ID", "Temperature", "Humidity"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_worksheet_titles() -> list[str]:
    """
    List worksheet/tab names via the underlying gspread client.
    This uses internal attributes of st-gsheets-connection, but is stable in practice.
    """
    client = conn.client  # underlying GSheetsServiceAccountClient when using service_account
    # Open spreadsheet using the URL in secrets
    ss = client._open_spreadsheet(spreadsheet=SPREADSHEET)  # private helper in library
    return [ws.title for ws in ss.worksheets()]


# ---------- Sidebar: pick worksheet tab ----------
st.sidebar.header("Google Sheet tabs")

try:
    titles = get_worksheet_titles()
except Exception as e:
    st.sidebar.error(f"Could not list worksheet tabs: {e}")
    titles = ["Sheet1"]

selected_ws = st.sidebar.radio(
    "Select worksheet (tab)",
    options=titles,
    index=0 if "Sheet1" not in titles else titles.index("Sheet1"),
)

refresh = st.sidebar.button("🔄 Refresh", type="secondary")
if refresh:
    st.rerun()

# ---------- Load from selected worksheet ----------
# Use a short ttl so changes appear quickly; gsheets-connection supports ttl on read(). [page:1]
df_raw = conn.read(worksheet=selected_ws, ttl=5)

# If the tab is empty, create a starter frame
if df_raw is None or df_raw.empty:
    df_raw = pd.DataFrame(
        {
            "Row": [1, 2, 3, 4],
            "Time": ["10:00", "", "", ""],
            "Sensor ID": [1, 2, 3, 4],
            "Temperature": [20.5, 21.0, 19.5, 20.2],
            "Humidity": [45, 46, 50, 44],
        }
    )

df_raw = normalize_columns(df_raw)
df_raw = ensure_row_column(df_raw)

# ---------- Main UI ----------
st.title("📊 Sensor Analytics (Cloud)")
st.caption(f"Active worksheet: {selected_ws}")

c_left, c_right = st.columns([4, 1])
with c_left:
    st.subheader("📝 Data Editor")
    st.caption("Row IDs are auto-managed. Time blanks are forward-filled for analysis.")
with c_right:
    save_clicked = st.button("💾 Save", type="primary")  # keep simple; no deprecated width args

edited_df = st.data_editor(
    df_raw,
    num_rows="dynamic",
    width="stretch",
    key=f"editor_{selected_ws}",
    column_config={
        "Row": st.column_config.NumberColumn("Row", help="Auto-generated unique row id", disabled=True),
    },
)

# Save: ensure Row IDs exist before writing
if save_clicked:
    try:
        to_save = normalize_columns(edited_df)
        to_save = ensure_row_column(to_save)
        # Write back to the chosen worksheet. update() clears then writes the dataframe. [page:1]
        conn.update(worksheet=selected_ws, data=to_save)
        st.success("Saved to Google Sheets.")
    except Exception as e:
        st.error(f"Save failed: {e}")

# ---------- Build plots ----------
try:
    df = process_data(ensure_row_column(normalize_columns(edited_df)))
    plot_df = df.dropna(subset=["Datetime", "Sensor ID"]).sort_values(["Datetime", "Sensor ID"])
except Exception as e:
    st.error(f"Processing error: {e}")
    plot_df = pd.DataFrame()

if plot_df.empty:
    st.warning("No valid data to visualize (check Time + Sensor ID columns).")
    st.stop()

def configure_chart(fig):
    fig.update_xaxes(tickformat="%H:%M", showgrid=True)
    return fig

st.divider()
st.subheader("📈 Network averages")

avg_df = plot_df.groupby("Datetime")[["Temperature", "Humidity"]].mean().reset_index()
col1, col2 = st.columns(2)

fig_avg_temp = px.area(avg_df, x="Datetime", y="Temperature", title="Avg Temperature (°C)", markers=True)
col1.plotly_chart(configure_chart(fig_avg_temp), width="stretch", key=f"avg_temp_{selected_ws}")

fig_avg_hum = px.area(avg_df, x="Datetime", y="Humidity", title="Avg Humidity (%)", markers=True)
col2.plotly_chart(configure_chart(fig_avg_hum), width="stretch", key=f"avg_hum_{selected_ws}")

st.divider()
st.subheader("🔍 Individual sensors")

sensors = sorted([s for s in plot_df["Sensor ID"].unique() if pd.notna(s)])
tabs = st.tabs([f"Sensor {int(s)}" for s in sensors])

for tab, sid in zip(tabs, sensors):
    with tab:
        s_data = plot_df[plot_df["Sensor ID"] == sid]
        a, b = st.columns(2)

        fig_t = px.line(s_data, x="Datetime", y="Temperature", title="Temperature", markers=True)
        a.plotly_chart(configure_chart(fig_t), width="stretch", key=f"temp_{selected_ws}_{sid}")

        fig_h = px.line(s_data, x="Datetime", y="Humidity", title="Humidity", markers=True)
        b.plotly_chart(configure_chart(fig_h), width="stretch", key=f"hum_{selected_ws}_{sid}")
