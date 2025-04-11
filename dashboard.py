import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import plotly.express as px # Import Plotly Express for histogram

st.set_page_config(page_title="LoL Team Objectives Dashboard", layout="wide")

# --- 1. Updated Password Check using st.secrets ---
def check_password():
    """Returns True if the user has entered the correct password."""
    def password_entered():
        """Checks if the entered password matches the one in secrets."""
        if st.session_state["password"] == st.secrets["izi_password"]: # Use st.secrets
            st.session_state["authentication_status"] = True
            del st.session_state["password"]  # Clear the password from session state
        else:
            st.session_state["authentication_status"] = False
            st.error("Incorrect password") # Provide immediate feedback

    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None

    # Removed the redundant error message here, handled in password_entered

    if st.session_state["authentication_status"] is True:
        return True
    else:
        # Show password input only if not authenticated
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        if st.session_state["authentication_status"] is False:
             st.error("Incorrect password. Please try again.") # Persistent error if failed
        st.stop() # Stop execution if not authenticated
    return False # Should not be reached if st.stop() works

if check_password():
    st.title("LoL Team Objectives Dashboard")
    # Welcome message removed for cleaner look once logged in

    # Load data
    @st.cache_data
    def load_data():
        try:
            df = pd.read_excel("data/objectives_db.xlsx", parse_dates=["date"])
        except FileNotFoundError:
            st.error("Error: 'data/objectives_db.xlsx' not found. Please make sure the file exists.")
            st.stop()

        def convert_time_to_seconds(time_str):
            if pd.isna(time_str): return np.nan
            try:
                # Handle potential float inputs from Excel (e.g., time stored as fraction of a day)
                if isinstance(time_str, (float, int)):
                     # Assuming float represents fraction of a day if it's like 0.xxxxx
                     if 0 < time_str < 1:
                         total_seconds = time_str * 24 * 60 * 60
                         return total_seconds
                     else: # Or if it's just seconds already
                         return float(time_str)
                # Handle "MM:SS" string format
                minutes, seconds = map(int, str(time_str).split(':'))
                return minutes * 60 + seconds
            except (ValueError, AttributeError, TypeError):
                return np.nan # Return NaN for any conversion error

        # Apply time conversion robustly
        df["timing_seconds"] = df["timing"].apply(convert_time_to_seconds)
        # Keep original timing for display if needed, or drop if only seconds are used
        # df = df.drop(columns=['timing'])

        # --- Data Validation ---
        required_cols = ['date', 'Blue', 'Red', 'objective', 'taken', 'fought', 'timing_seconds',
                         'blue_top', 'blue_jungle', 'blue_mid', 'blue_AD', 'blue_sup',
                         'red_top', 'red_jungle', 'red_mid', 'red_AD', 'red_sup']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Error: The following required columns are missing from the data: {', '.join(missing_cols)}")
            st.stop()

        # Ensure 'taken' and 'fought' columns exist for later logic
        if 'taken' not in df.columns or 'fought' not in df.columns:
             st.error("Error: 'taken' and 'fought' columns are essential for analysis and are missing.")
             st.stop()

        return df

    df_raw = load_data()

    st.sidebar.header("Filters")

    # --- Date Filter ---
    min_date = df_raw['date'].min().date()
    max_date = df_raw['date'].max().date()
    selected_date_range = st.sidebar.date_input(
        "Select Date Range",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date # Use min/max_value for better picker constraints
    )
    start_date, end_date = selected_date_range

    # Filter data by date first for efficiency
    df_filtered_date = df_raw[(df_raw['date'].dt.date >= start_date) & (df_raw['date'].dt.date <= end_date)].copy()

    if df_filtered_date.empty:
        st.warning("No data available for the selected date range.")
        st.stop()

    # --- Side Filter ---
    side_options = ["All Sides", "Blue", "Red"]
    side = st.sidebar.radio("Select Side", side_options)

    # --- Team Filter ---
    if side == "Blue":
        team_options = sorted(df_filtered_date['Blue'].unique())
        opponent_col_name = 'Red'
        team_col_name = 'Blue'
    elif side == "Red":
        team_options = sorted(df_filtered_date['Red'].unique())
        opponent_col_name = 'Blue'
        team_col_name = 'Red'
    else: # All Sides
        team_options = sorted(list(pd.concat([df_filtered_date['Blue'], df_filtered_date['Red']]).unique()))
        opponent_col_name = None # Opponent filter more complex for 'All Sides'
        team_col_name = None

    if not team_options:
         st.warning("No teams found in the selected date range.")
         st.stop()

    selected_team = st.sidebar.selectbox("Select Team", team_options)

    # --- 4. Opponent Filter ---
    opponent_options = ["All Opponents"]
    if selected_team and opponent_col_name: # Only show if a team and specific side are selected
        opponent_options.extend(sorted(df_filtered_date[df_filtered_date[team_col_name] == selected_team][opponent_col_name].unique()))
    elif selected_team and side == "All Sides":
         # Find all opponents this team played against on either side
         opponents_vs_blue = df_filtered_date[df_filtered_date['Blue'] == selected_team]['Red'].unique()
         opponents_vs_red = df_filtered_date[df_filtered_date['Red'] == selected_team]['Blue'].unique()
         all_opps = set(opponents_vs_blue) | set(opponents_vs_red)
         opponent_options.extend(sorted(list(all_opps)))

    selected_opponent = st.sidebar.selectbox("Select Opponent", opponent_options, disabled=(not selected_team))


    # --- Objective Filter ---
    objective_options = sorted(df_filtered_date['objective'].unique())
    # Sensible default: select first objective, or None if list is empty
    default_objective = [objective_options[0]] if objective_options else None
    selected_objectives = st.sidebar.multiselect(
        "Select Objectives",
        objective_options,
        default=default_objective
    )

    # --- Drake Type Filter (Conditional) ---
    drake_objectives = ['infernal_drake', 'ocean_drake', 'mountain_drake', 'cloud_drake', 'hextech_drake', 'chemtech_drake', 'elder_drake']
    is_drake_objective_selected = any(obj in drake_objectives for obj in selected_objectives)
    selected_drake_types = ['All'] # Default value
    if is_drake_objective_selected:
        all_drake_types = df_filtered_date['drake_type'].dropna().unique()
        # Ensure 'All' is an option even if no drakes are in the filtered data yet
        drake_filter_options = ['All'] + sorted(list(all_drake_types))
        selected_drake_types = st.sidebar.multiselect(
            "Select Drake Types",
            drake_filter_options,
            default=['All']
        )

    # --- Apply Filters ---
    # Start with date-filtered data
    df_display = df_filtered_date.copy()

    # Filter by Team and Side
    if selected_team:
        if side == "Blue":
            df_display = df_display[df_display['Blue'] == selected_team]
        elif side == "Red":
            df_display = df_display[df_display['Red'] == selected_team]
        elif side == "All Sides":
            df_display = df_display[(df_display['Blue'] == selected_team) | (df_display['Red'] == selected_team)]

    # Filter by Opponent
    if selected_opponent != "All Opponents" and selected_team:
        if side == "Blue":
            df_display = df_display[df_display['Red'] == selected_opponent]
        elif side == "Red":
            df_display = df_display[df_display['Blue'] == selected_opponent]
        elif side == "All Sides":
             # Match games where the selected team played the selected opponent, regardless of side
            condition1 = (df_display['Blue'] == selected_team) & (df_display['Red'] == selected_opponent)
            condition2 = (df_display['Red'] == selected_team) & (df_display['Blue'] == selected_opponent)
            df_display = df_display[condition1 | condition2]

    # Filter by Objectives
    if selected_objectives:
        df_display = df_display[df_display['objective'].isin(selected_objectives)]

    # Filter by Drake Type (if applicable)
    if is_drake_objective_selected and 'All' not in selected_drake_types:
        # Filter only the rows that are drake objectives *and* match the selected types
        drake_rows_mask = df_display['objective'].isin(drake_objectives)
        type_match_mask = df_display['drake_type'].isin(selected_drake_types)
        # Keep non-drake rows OR drake rows that match the type
        df_display = df_display[~drake_rows_mask | (drake_rows_mask & type_match_mask)]

    # --- Dynamic Title ---
    title_string = f"{selected_team}"
    if side != "All Sides":
        title_string += f" ({side} Side)"
    if selected_opponent != "All Opponents":
        title_string += f" vs {selected_opponent}"
    title_string += " Performance"
    if selected_objectives:
        title_string += f" on {', '.join(selected_objectives)}"
        if is_drake_objective_selected and 'All' not in selected_drake_types:
             title_string += f" ({', '.join(selected_drake_types)})"
    st.title(title_string)

    # --- Main Analysis Area ---
    if not df_display.empty:
        total_objectives_count = len(df_display)

        st.subheader("Key Objective Metrics")
        col1, col2, col3, col4 = st.columns(4)

        # --- Metric: Contest Rate ---
        # Assumes 'fought' column contains selected_team name, opponent name, "Both", or NaN/Other
        # Count instances where the selected team was involved in a contest
        fought_mask = df_display['fought'].isin([selected_team, "Both"])
        fought_count = fought_mask.sum()
        percentage_fought = (fought_count / total_objectives_count * 100) if total_objectives_count > 0 else 0
        # --- 6. Rename Metric ---
        col1.metric("Contest Rate", f"{percentage_fought:.1f}%", help="Percentage of these objectives where the selected team was involved in a contest ('fought' = team name or 'Both').")

        # --- Metric: Average Timing (Fought) ---
        fought_timings = df_display.loc[fought_mask, 'timing_seconds'].dropna()
        if not fought_timings.empty:
            avg_timing_seconds = fought_timings.mean()
            avg_timing_minutes = int(avg_timing_seconds // 60)
            avg_timing_seconds_rem = int(avg_timing_seconds % 60)
            col2.metric("Avg Contest Timing", f"{avg_timing_minutes:02d}:{avg_timing_seconds_rem:02d}", help="Average game time when the selected team contested the objective.")
        else:
            col2.metric("Avg Contest Timing", "N/A", help="No contested instances with timing data.")

        # --- Metric: Percentage Taken (by selected team) ---
        taken_mask = (df_display['taken'] == selected_team)
        taken_count = taken_mask.sum()
        percentage_taken = (taken_count / total_objectives_count * 100) if total_objectives_count > 0 else 0
        col3.metric("Taken Rate", f"{percentage_taken:.1f}%", help="Percentage of these objectives secured by the selected team.")

        # --- 2. Metric: Objective Control Rate (OCR) ---
        # Requires knowing opponent's name when they take it
        # Assumption: 'taken' contains the name of the team who took it.
        # Find instances taken by *anyone* other than the selected team (could be opponent or neither if data allows)
        # More robustly: count objectives taken by the selected team vs objectives taken by the opponent *in games selected team played*
        opponent_team_name = None
        if selected_opponent != "All Opponents":
            opponent_team_name = selected_opponent
        elif side == "Blue" and not df_display.empty:
            opponent_team_name = df_display['Red'].iloc[0] # Get opponent from first row (assuming consistent opponent if not 'All')
        elif side == "Red" and not df_display.empty:
            opponent_team_name = df_display['Blue'].iloc[0]
        # If side is "All Sides" and opponent is "All Opponents", OCR is less clearly defined for the whole set.
        # We calculate OCR based on *decided* objectives (taken by selected team OR assumed opponent)

        taken_by_opponent_count = 0
        if opponent_team_name: # If we have a single opponent context
            taken_by_opponent_count = (df_display['taken'] == opponent_team_name).sum()
        else: # If 'All Opponents', sum takes by anyone *not* the selected team
             taken_by_opponent_count = (~taken_mask & df_display['taken'].notna() & (df_display['taken'] != '')).sum()


        total_decided_objectives = taken_count + taken_by_opponent_count
        ocr = (taken_count / total_decided_objectives * 100) if total_decided_objectives > 0 else 0
        col4.metric("Objective Control Rate (OCR)", f"{ocr:.1f}%", help="Percentage of objectives secured by the selected team out of all objectives secured by either the selected team or the opponent(s).")

        # --- 3. Timing Distribution Plot ---
        st.subheader("Timing Distribution (Taken by Selected Team)") # Clarified title
        taken_mask = (df_display['taken'] == selected_team) # Re-ensure mask is defined here
        taken_timings_df = df_display.loc[taken_mask, 'timing_seconds'].dropna()

        if not taken_timings_df.empty:
            # --- Helper function for formatting ---
            def format_seconds(sec):
                if pd.isna(sec) or not isinstance(sec, (int, float)): return "N/A"
                minutes = int(sec // 60)
                seconds = int(sec % 60)
                return f"{minutes:02d}:{seconds:02d}"

            # --- Create Histogram ---
            fig_hist = px.histogram(
                x=taken_timings_df,
                nbins=max(10, int(len(taken_timings_df)**0.5)), # Auto-adjust bins
            )

            # --- Customize X-axis Ticks ---
            min_time_sec = 0 # Start axis at 00:00
            max_time_sec = taken_timings_df.max()
            # Determine a reasonable tick interval (e.g., every 60s, 120s, 300s)
            if max_time_sec <= 300: # Up to 5 mins
                tick_interval_sec = 30
            elif max_time_sec <= 900: # Up to 15 mins
                tick_interval_sec = 60
            elif max_time_sec <= 1800: # Up to 30 mins
                tick_interval_sec = 120 # Every 2 mins
            elif max_time_sec <= 3600: # Up to 60 mins
                 tick_interval_sec = 300 # Every 5 mins
            else: # Very long games
                 tick_interval_sec = 600 # Every 10 mins

            # Generate tick values (positions in seconds)
            tickvals = np.arange(min_time_sec, max_time_sec + tick_interval_sec, tick_interval_sec)
            # Generate corresponding MM:SS labels
            ticktext = [format_seconds(val) for val in tickvals]

            # --- Customize Hover Data ---
            # The default hover shows the bin range. We'll try to add the formatted center/start.
            # This is an approximation as hover data for histograms is complex.
            fig_hist.update_traces(
                 hovertemplate="<b>Timing Bin:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
                 # Note: %{x} in histogram hover often refers to the bin range, not a single value.
                 # Getting precise MM:SS in hover for the *range* requires more advanced callbacks or manipulation.
                 # For simplicity, this template shows the raw second range. Let's update layout title instead.
            )


            # --- Update Layout ---
            fig_hist.update_layout(
                xaxis_title="Timing (MM:SS)", # Updated axis title
                yaxis_title="Number of Objectives Taken",
                bargap=0.1,
                xaxis=dict(
                    tickmode='array',
                    tickvals=tickvals,
                    ticktext=ticktext
                ),
                hoverlabel=dict( # Improve hover label appearance
                    bgcolor="white",
                    font_size=12,
                    # font_family="Rockwell" # Example font
                )
            )

            # --- Add Mean Line with Formatted Annotation ---
            avg_taken_timing_sec = taken_timings_df.mean()
            avg_taken_timing_formatted = format_seconds(avg_taken_timing_sec)
            fig_hist.add_vline(
                 x=avg_taken_timing_sec,
                 line_dash="dash",
                 line_color="red",
                 annotation_text=f"Mean: {avg_taken_timing_formatted}",
                 annotation_position="top right" # Position annotation
                 )

            st.plotly_chart(fig_hist, use_container_width=True)
        else:
             st.info("No objectives taken by the selected team with timing data to plot distribution.")

        # --- Player Presence Analysis ---
        st.subheader("Average Player Presence (During Objective Instance)")
        presence_data = {}
        # Determine relevant player columns based on side and team selection
        player_cols_map = {}
        if side == "Blue":
            player_cols_map = {"Top": "blue_top", "Jungle": "blue_jungle", "Mid": "blue_mid", "AD": "blue_AD", "Sup": "blue_sup"}
        elif side == "Red":
            player_cols_map = {"Top": "red_top", "Jungle": "red_jungle", "Mid": "red_mid", "AD": "red_AD", "Sup": "red_sup"}
        elif side == "All Sides": # Need to check both sides
             player_cols_map = {
                 "Top": ["blue_top", "red_top"], "Jungle": ["blue_jungle", "red_jungle"],
                 "Mid": ["blue_mid", "red_mid"], "AD": ["blue_AD", "red_AD"], "Sup": ["blue_sup", "red_sup"]
             }


        if selected_team and player_cols_map:
            all_roles_presence = {}
            role_names = ["Top", "Jungle", "Mid", "AD", "Sup"]

            for role in role_names:
                total_presence_count = 0
                current_total_objectives = 0 # Recalculate denominator per role if using 'All Sides'

                if side in ["Blue", "Red"]:
                     col_name = player_cols_map[role]
                     if col_name in df_display.columns:
                         # Check for 'X' (or 1, True depending on data) indicating presence
                         # Making presence check more robust: == 'X' or == 1 or == True
                         presence_check = (df_display[col_name] == 'X') | (df_display[col_name] == 1) | (df_display[col_name] == True)
                         total_presence_count = df_display[presence_check].shape[0]
                         current_total_objectives = total_objectives_count # Use overall count for Blue/Red
                elif side == "All Sides":
                     blue_col, red_col = player_cols_map[role]
                     # Count presence on blue side games *for selected team*
                     blue_df_role = df_display[(df_display['Blue'] == selected_team)]
                     if blue_col in blue_df_role.columns:
                          presence_check_blue = (blue_df_role[blue_col] == 'X') | (blue_df_role[blue_col] == 1) | (blue_df_role[blue_col] == True)
                          total_presence_count += blue_df_role[presence_check_blue].shape[0]
                          current_total_objectives += len(blue_df_role)
                     # Count presence on red side games *for selected team*
                     red_df_role = df_display[(df_display['Red'] == selected_team)]
                     if red_col in red_df_role.columns:
                          presence_check_red = (red_df_role[red_col] == 'X') | (red_df_role[red_col] == 1) | (red_df_role[red_col] == True)
                          total_presence_count += red_df_role[presence_check_red].shape[0]
                          current_total_objectives += len(red_df_role) # Add red side games count


                # Use the appropriate denominator
                denominator = current_total_objectives if side == "All Sides" else total_objectives_count
                avg_presence_percentage = (total_presence_count / denominator * 100) if denominator > 0 else 0
                # Ensure percentage doesn't exceed 100 (though it shouldn't mathematically)
                all_roles_presence[role] = min(avg_presence_percentage, 100.0)

            if all_roles_presence:
                # Display metrics
                num_cols = len(all_roles_presence)
                cols_pres = st.columns(num_cols)
                roles_list = list(all_roles_presence.keys())
                presence_values = list(all_roles_presence.values())
                for i, role in enumerate(roles_list):
                    cols_pres[i].metric(f"{role} Presence", f"{all_roles_presence[role]:.1f}%", help=f"Average presence of {selected_team}'s {role} player during these objective instances.")

                # Radar Chart
                # Ensure list isn't empty before trying to access index 0
                if presence_values:
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=presence_values + [presence_values[0]], # Close the loop
                        theta=roles_list + [roles_list[0]],      # Close the loop
                        fill='toself',
                        name=selected_team
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100] # Set fixed range 0-100%
                            )),
                        showlegend=False,
                        title="Player Presence Pattern"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("No presence data available to display radar chart.")


        # --- Data Table & Download ---
        st.subheader("Filtered Data")

        # --- 5. Download Button ---
        @st.cache_data # Cache the conversion to CSV
        def convert_df_to_csv(df_to_convert):
           # IMPORTANT: Cache the conversion to prevent computation on every rerun
           return df_to_convert.to_csv(index=False).encode('utf-8')

        csv_data = convert_df_to_csv(df_display)

        st.download_button(
           label="Download Data as CSV",
           data=csv_data,
           file_name=f"{selected_team}_objectives_{start_date}_to_{end_date}.csv",
           mime='text/csv',
           key='download-csv' # Added a key for stability
        )

        # Display the dataframe
        st.dataframe(df_display, use_container_width=True)

    else:
        # Displayed if df_display is empty after filtering
        st.warning("No data available for the selected filters. Try adjusting the date range, team, objectives, or opponent.")

