import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np

# For Plotly Radar Chart
import plotly.graph_objects as go

st.set_page_config(page_title="LoL Team Objectives Dashboard", layout="wide")

def check_password():
    """Returns True if the user has entered the correct password."""
    def password_entered():
        """Checks if the entered password matches the hardcoded password."""
        if st.session_state["password"] == "izispring25":
            st.session_state["authentication_status"] = True
            del st.session_state["password"]  # Clear the password from session state
        else:
            st.session_state["authentication_status"] = False

    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None  # Set initial state

    if st.session_state["authentication_status"] is False:
        st.error("Incorrect password")
    elif st.session_state["authentication_status"] is True:
        return True

    # Show password input
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    return False

if check_password():
    # Your main Streamlit application code goes here
    st.title("LoL Team Objectives Dashboard")
    st.write("Welcome to the dashboard!")

    # Load data
    @st.cache_data
    def load_data():
        # Assumes 'objectives_db.xlsx' is placed in the 'data' folder
        df = pd.read_excel("data/objectives_db.xlsx", parse_dates=["date"])
        # Convert timing from "MM:SS" format to seconds
        def convert_time_to_seconds(time_str):
            if pd.isna(time_str):
                return np.nan
            try:
                minutes, seconds = map(int, time_str.split(':'))
                return minutes * 60 + seconds
            except (ValueError, AttributeError):
                return np.nan

        df["timing"] = df["timing"].apply(convert_time_to_seconds)
        return df

    df = load_data()

    st.sidebar.header("Filters")

    # Date Filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    selected_date_range = st.sidebar.date_input("Select Date Range", (min_date, max_date), min_date, max_date)

    side_options = ["All Sides", "Blue", "Red"]
    side = st.sidebar.radio("Select Side", side_options)

    if side == "Blue":
        team_options = df['Blue'].unique()
        team_col = 'Blue'
        opponent_col = 'Red'
        player_cols = ["blue_top", "blue_jungle", "blue_mid", "blue_AD", "blue_sup"]
        fought_value = selected_team if 'selected_team' in locals() else None
        taken_value = selected_team if 'selected_team' in locals() else None
        fought_both_value = "Both"
    elif side == "Red":
        team_options = df['Red'].unique()
        team_col = 'Red'
        opponent_col = 'Blue'
        player_cols = ["red_top", "red_jungle", "red_mid", "red_AD", "red_sup"]
        fought_value = selected_team if 'selected_team' in locals() else None
        taken_value = selected_team if 'selected_team' in locals() else None
        fought_both_value = "Both"
    else:  # side == "All Sides"
        team_options = sorted(list(pd.concat([df['Blue'], df['Red']]).unique()))
        team_col = None
        opponent_col = None
        player_cols = ["blue_top", "blue_jungle", "blue_mid", "blue_AD", "blue_sup",
                       "red_top", "red_jungle", "red_mid", "red_AD", "red_sup"]
        fought_value = None
        taken_value = None
        fought_both_value = "Both"

    selected_team = st.sidebar.selectbox("Select Team", team_options)

    if side == "Blue":
        fought_value = selected_team
        taken_value = selected_team
    elif side == "Red":
        fought_value = selected_team
        taken_value = selected_team

    objective_options = df['objective'].unique()
    selected_objectives = st.sidebar.multiselect("Select Objectives", objective_options, default=objective_options[0] if len(objective_options) > 0 else None)

    # Drake Type Filter (only if at least one drake objective is selected)
    drake_objectives = ['infernal_drake', 'ocean_drake', 'mountain_drake', 'cloud_drake', 'hextech_drake', 'chemtech_drake', 'elder_drake']
    is_drake_objective_selected = any(obj in drake_objectives for obj in selected_objectives)
    if is_drake_objective_selected:
        all_drake_types = df['drake_type'].dropna().unique()
        selected_drake_types = st.sidebar.multiselect("Select Drake Types", ['All'] + list(all_drake_types), default=['All'])
    else:
        selected_drake_types = ['All']

    title_string = f"{selected_team} Performance on Selected Objectives"
    if side != "All Sides":
        title_string = f"{selected_team} ({side} Side) Performance on Selected Objectives"
    st.title(title_string)

    # Filter data based on selected date range
    start_date, end_date = selected_date_range
    filtered_df_date = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)].copy()

    # Filter data based on selected side, team, and objectives
    if side == "Blue" and selected_team:
        filtered_df = filtered_df_date[(filtered_df_date['Blue'] == selected_team) & (filtered_df_date['objective'].isin(selected_objectives))].copy()
    elif side == "Red" and selected_team:
        filtered_df = filtered_df_date[(filtered_df_date['Red'] == selected_team) & (filtered_df_date['objective'].isin(selected_objectives))].copy()
    elif side == "All Sides" and selected_team:
        filtered_df_blue = filtered_df_date[(filtered_df_date['Blue'] == selected_team) & (filtered_df_date['objective'].isin(selected_objectives))].copy()
        filtered_df_red = filtered_df_date[(filtered_df_date['Red'] == selected_team) & (filtered_df_date['objective'].isin(selected_objectives))].copy()
        filtered_df = pd.concat([filtered_df_blue, filtered_df_red])
    else:
        filtered_df = pd.DataFrame()

    # Apply Drake Type Filter
    if is_drake_objective_selected and 'All' not in selected_drake_types and not filtered_df.empty:
        filtered_df = filtered_df[filtered_df['drake_type'].isin(selected_drake_types)]

    if not filtered_df.empty:
        subheader_string = f"Analysis for {selected_team} on {', '.join(selected_objectives)}"
        if is_drake_objective_selected and 'All' not in selected_drake_types:
            subheader_string += f" ({', '.join(selected_drake_types)})"
        subheader_string += f" for the period {start_date} to {end_date}"
        st.subheader(subheader_string)

        total_objectives_count = len(filtered_df)
        st.metric("Total Instances", total_objectives_count)

        # Percentage of time fought (considering "Both")
        fought_count = 0
        fought_timings = pd.Series([])
        if side == "Blue" and selected_team:
            fought_df = filtered_df[filtered_df['fought'].isin([selected_team, fought_both_value])]
            fought_count = fought_df.shape[0]
            fought_timings = fought_df['timing'].dropna()
        elif side == "Red" and selected_team:
            fought_df = filtered_df[filtered_df['fought'].isin([selected_team, fought_both_value])]
            fought_count = fought_df.shape[0]
            fought_timings = fought_df['timing'].dropna()
        elif side == "All Sides" and selected_team:
            fought_df_blue = filtered_df_blue[filtered_df_blue['fought'].isin([selected_team, fought_both_value])]
            fought_df_red = filtered_df_red[filtered_df_red['fought'].isin([selected_team, fought_both_value])]
            fought_count = fought_df_blue.shape[0] + fought_df_red.shape[0]
            fought_timings = pd.concat([fought_df_blue['timing'].dropna(), fought_df_red['timing'].dropna()])

        percentage_fought = min((fought_count / total_objectives_count) * 100 if total_objectives_count > 0 else 0, 100)
        st.metric("Percentage Fought", f"{percentage_fought:.2f}%")

        # Average Timing (Fought)
        if not fought_timings.empty:
            avg_timing_seconds = fought_timings.mean()
            avg_timing_minutes = int(avg_timing_seconds // 60)
            avg_timing_seconds_rem = int(avg_timing_seconds % 60)
            st.metric("Average Timing (Fought)", f"{avg_timing_minutes:02d}:{avg_timing_seconds_rem:02d}")
        else:
            st.metric("Average Timing (Fought)", "No instances fought")

        # Percentage of time taken
        taken_count = 0
        if side == "Blue" and selected_team:
            taken_count = filtered_df[filtered_df['taken'] == selected_team].shape[0]
        elif side == "Red" and selected_team:
            taken_count = filtered_df[filtered_df['taken'] == selected_team].shape[0]
        elif side == "All Sides" and selected_team:
            taken_count_blue = filtered_df_blue[filtered_df_blue['taken'] == selected_team].shape[0]
            taken_count_red = filtered_df_red[filtered_df_red['taken'] == selected_team].shape[0]
            taken_count = taken_count_blue + taken_count_red

        percentage_taken = min((taken_count / total_objectives_count) * 100 if total_objectives_count > 0 else 0, 100)
        st.metric("Percentage Taken", f"{percentage_taken:.2f}%")

        st.subheader("Average Player Presence")
        presence_data = {}
        if selected_team and player_cols:
            all_roles_presence = {}
            role_names = ["Top", "Jungle", "Mid", "AD", "Sup"]

            for i, role in enumerate(role_names):
                blue_col = f"blue_{role.lower()}"
                red_col = f"red_{role.lower()}"
                total_presence_count = 0

                if side in ["Blue", "All Sides"] and blue_col in filtered_df.columns:
                    blue_df = filtered_df[filtered_df['Blue'] == selected_team] if side == "Blue" else filtered_df_blue
                    total_presence_count += blue_df[blue_df[blue_col] == 'X'].shape[0]
                if side in ["Red", "All Sides"] and red_col in filtered_df.columns:
                    red_df = filtered_df[filtered_df['Red'] == selected_team] if side == "Red" else filtered_df_red
                    total_presence_count += red_df[red_df[red_col] == 'X'].shape[0]

                avg_presence_percentage = min((total_presence_count / total_objectives_count) * 100 if total_objectives_count > 0 else 0, 100)
                all_roles_presence[role] = avg_presence_percentage

            if all_roles_presence:
                num_cols = len(all_roles_presence)
                cols = st.columns(num_cols)
                roles = list(all_roles_presence.keys())
                presence_values = list(all_roles_presence.values())
                for i, role in enumerate(roles):
                    cols[i].metric(f"{selected_team} {role} Presence (Avg)", f"{all_roles_presence[role]:.2f}%")

                # --- Plotly Radar Chart ---
                st.subheader("Player Presence Radar Chart")
                fig = go.Figure(data=[go.Scatterpolar(
                    r=presence_values + [presence_values[0]],
                    theta=roles + [roles[0]],
                    fill='toself',
                    name=selected_team
                )])

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data Table")
        st.dataframe(filtered_df)

    else:
        st.info("No data available for the selected team and objective(s) within the selected date range.")