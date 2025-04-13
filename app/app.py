import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules
from src.data_processor import DataProcessor
from src.batter_analyzer import BatterVulnerabilityAnalyzer
from src.bowler_analyzer import BowlerAnalyzer 
from src.bowling_plan_generator import BowlingPlanGenerator
from utils.visualization import (
    create_vulnerability_heatmap,
    create_field_placement_visualization,
    get_strategy_visualization,
    create_bowler_economy_heatmap,
    create_bowler_strike_rate_heatmap
)

# Page configuration
st.set_page_config(
    page_title="T20 Bowling Strategy Analyzer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.data = None
    st.session_state.batter_analyzer = None
    st.session_state.bowler_analyzer = None
    st.session_state.plan_generator = None
    st.session_state.current_page = "Batter Analysis"
    st.session_state.app_started = False
    st.session_state.data_loaded = False
    st.session_state.analyzers_initialized = False

# App title and description
st.title("🏏 T20 Bowling Strategy Analyzer")
st.markdown("""
This application helps cricket teams develop data-driven bowling strategies 
for T20 matches based on ball-by-ball analysis.
""")

# Start button
if not st.session_state.app_started:
    if st.button("Start Application"):
        st.session_state.app_started = True
        st.rerun()
    st.stop()

# Sidebar for data input and global controls
with st.sidebar:
    st.header("Data Input")
    
    if not st.session_state.data_loaded:
        # Load local data file
        data_path = Path(__file__).parent.parent / "data" / "t20_bbb.csv"
        if data_path.exists():
            try:
                with st.spinner("Loading data file..."):
                    # Load data with low_memory=False to handle mixed types
                    data = pd.read_csv(data_path, low_memory=False)
                    st.success("Data file loaded successfully")
                
                with st.spinner("Validating data..."):
                    # Ensure required columns exist
                    required_columns = ['p_match', 'bat', 'bowl', 'line', 'length', 'phase', 'score', 'out']
                    missing_columns = [col for col in required_columns if col not in data.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    else:
                        st.success("Data validation successful")
                        
                        with st.spinner("Processing data..."):
                            try:
                                # Process the data
                                data_processor = DataProcessor(data)
                                processed_data = data_processor.process()
                                st.success("Data processing completed")
                                
                                # Store in session state
                                st.session_state.data = processed_data
                                st.session_state.data_loaded = True
                                
                                # Display data stats
                                st.subheader("Dataset Statistics")
                                st.write(f"Matches: {processed_data['p_match'].nunique()}")
                                st.write(f"Batters: {processed_data['bat'].nunique()}")
                                st.write(f"Bowlers: {processed_data['bowl'].nunique()}")
                                
                                # Initialize analyzers with progress tracking
                                st.write("Initializing analyzers...")
                                progress_bar = st.progress(0)
                                
                                st.write("Creating BatterVulnerabilityAnalyzer...")
                                try:
                                    # Initialize with required columns
                                    required_cols = ['bat', 'bat_hand', 'score', 'out', 'bowl_kind']
                                    optional_cols = ['phase', 'line', 'length']
                                    analysis_cols = required_cols + [col for col in optional_cols if col in processed_data.columns]
                                    analysis_data = processed_data[analysis_cols].copy()
                                    st.session_state.batter_analyzer = BatterVulnerabilityAnalyzer(analysis_data)
                                    progress_bar.progress(25)
                                    st.success("Batter analyzer initialized")
                                    
                                    st.write("Creating BowlerAnalyzer...")
                                    # Initialize with only necessary columns for bowler analysis
                                    bowler_data = processed_data[['bowl', 'bowl_kind', 'bowl_style', 'line', 'length', 'score', 'out']].copy()
                                    st.session_state.bowler_analyzer = BowlerAnalyzer(bowler_data)
                                    progress_bar.progress(50)
                                    st.success("Bowler analyzer initialized")
                                    
                                    st.write("Creating BowlingPlanGenerator...")
                                    st.session_state.plan_generator = BowlingPlanGenerator(processed_data)
                                    progress_bar.progress(75)
                                    st.success("Plan generator initialized")
                                    
                                    progress_bar.progress(100)
                                    st.session_state.analyzers_initialized = True
                                    st.success("All analyzers initialized successfully!")
                                    
                                    time.sleep(1)  # Give users time to see the success message
                                    st.rerun()
                                    
                                except Exception as analyzer_error:
                                    st.error(f"Error initializing analyzers: {str(analyzer_error)}")
                                    st.error("Debug info:")
                                    st.write(f"Memory usage: {processed_data.memory_usage().sum() / 1024**2:.2f} MB")
                                    st.write(f"Number of rows: {len(processed_data)}")
                                    st.write(f"Columns: {', '.join(processed_data.columns)}")
                                    st.exception(analyzer_error)
                            except Exception as proc_error:
                                st.error(f"Error processing data: {str(proc_error)}")
                                st.exception(proc_error)

            except Exception as e:
                st.error(f"Error loading local data: {str(e)}")
                st.error("Please check the data format and try again.")
                st.exception(e)
        else:
            st.error("Local data file not found. Please ensure t20_bbb.csv exists in the data directory.")
    else:
        # Show navigation only after data is loaded
        st.markdown("---")
        st.header("Navigation")
        
        # Navigation options
        page = st.radio(
            "Select Analysis",
            ["Batter Analysis", "Bowler Strategies", "Match-up Optimization", "Complete Bowling Plan"],
            index=0,
            key="page_selection"
        )
        
        # Update current page in session state
        st.session_state.current_page = page

# Main content area based on selection
if not st.session_state.data_loaded:
    st.info("Please wait while the data is being loaded and processed...")
    st.stop()

# Update phase definitions where they appear
phase_names = {
    1: "Powerplay (Overs 1-6)",
    2: "Early Middle (Overs 7-12)",
    3: "Late Middle (Overs 13-16)", 
    4: "Death (Overs 17-20)"
}

# Render appropriate page based on selection
if st.session_state.current_page == "Batter Analysis":
    st.header("Batter Vulnerability Analysis")
    
    # Batter selection
    batters = sorted(st.session_state.data['bat'].unique())
    selected_batter = st.selectbox("Select Batter", batters)
    
    if selected_batter:
        # Get batter analysis
        with st.spinner("Analyzing batter..."):
            batter_profile = st.session_state.batter_analyzer.analyze_batter(selected_batter)
            
            if batter_profile:
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Batter Profile")
                    st.write(f"Batting Hand: {batter_profile.get('bat_hand', 'Unknown')}")
                    st.write(f"Strike Rate: {batter_profile.get('strike_rate', 0):.2f}")
                    
                    # Handle inf average
                    avg = batter_profile.get('average', 0)
                    avg_display = f"{avg:.2f}" if avg != float('inf') else "No dismissals"
                    st.write(f"Average: {avg_display}")
                    
                    st.write(f"Total Runs: {batter_profile.get('total_runs', 0)}")
                    st.write(f"Balls Faced: {batter_profile.get('total_balls', 0)}")
                    st.write(f"Dismissals: {batter_profile.get('dismissals', 0)}")
                    
                with col2:
                    st.subheader("Weakness Summary")
                    
                    if hasattr(st.session_state, 'plan_generator') and st.session_state.plan_generator:
                        weaknesses = st.session_state.plan_generator.find_batter_weaknesses(selected_batter)
                        
                        if weaknesses and weaknesses.get('weaknesses'):
                            for weakness in weaknesses.get('weaknesses', []):
                                if weakness['type'] == 'bowler_type':
                                    st.write(f"👉 Vulnerable against {weakness['bowler_type']}")
                                    st.write(f"   Average: {weakness['average']:.2f} (Overall: {weakness['overall_average']:.2f})")
                                    st.write(f"   Confidence: {weakness['confidence'].upper()}")
                                    
                                elif weakness['type'] == 'line_length':
                                    st.write(f"👉 Struggles against {weakness['line']} {weakness['length']} deliveries")
                                    st.write(f"   Average: {weakness['average']:.2f} (Overall: {weakness['overall_average']:.2f})")
                                    st.write(f"   Confidence: {weakness['confidence'].upper()}")
                                    
                                elif weakness['type'] == 'phase':
                                    phase = phase_names.get(weakness['phase'], weakness['phase'])
                                    st.write(f"👉 Less effective in {phase}")
                                    st.write(f"   Strike Rate: {weakness['strike_rate']:.2f} (Overall: {weakness['overall_strike_rate']:.2f})")
                                    st.write(f"   Confidence: {weakness['confidence'].upper()}")
                        else:
                            st.write("No significant weaknesses identified in the dataset.")
                    else:
                        st.write("Plan generator not initialized. Cannot analyze weaknesses.")
                
                # Create tabs for different analysis views
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Vulnerability Heatmap", 
                    "Performance by Bowler Type",
                    "Performance by Phase",
                    "Performance by Bowling Style"
                ])
                
                with tab1:
                    try:
                        heatmap_fig = create_vulnerability_heatmap(batter_profile)
                        st.pyplot(heatmap_fig)
                    except Exception as e:
                        st.error(f"Error creating vulnerability heatmap: {e}")
                        st.write("Not enough data to create a meaningful heatmap.")

                with tab2:
                    # Bar chart of performance against different bowler types
                    st.subheader("Performance vs Bowler Types")
                    
                    bowl_types = batter_profile.get('vs_bowler_types', {})
                    if bowl_types:
                        bowl_data = []
                        
                        for bowl_type, stats in bowl_types.items():
                            if stats['balls'] >= 5:  # Only include if sufficient data
                                bowl_data.append({
                                    'bowler_type': bowl_type,
                                    'strike_rate': stats['strike_rate'],
                                    'average': stats['average'] if stats['dismissals'] > 0 else None,
                                    'balls': stats['balls']
                                })
                        
                        if bowl_data:
                            bowl_df = pd.DataFrame(bowl_data)
                            
                            # Strike rate bar chart
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.bar(bowl_df['bowler_type'], bowl_df['strike_rate'], color='skyblue')
                            
                            # Add data labels
                            for i, bar in enumerate(bars):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                        f"{height:.1f}", ha='center', va='bottom', rotation=0)
                            
                            ax.set_title(f"{selected_batter}'s Strike Rate vs Different Bowler Types")
                            ax.set_xlabel("Bowler Type")
                            ax.set_ylabel("Strike Rate")
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            
                            # Display sample size for context
                            st.caption("Sample size (balls faced):")
                            for i, row in bowl_df.iterrows():
                                st.caption(f"{row['bowler_type']}: {row['balls']} balls")
                        else:
                            st.write("Not enough data to analyze performance against different bowler types.")
                    else:
                        st.write("No data available on performance against different bowler types.")
                        
                with tab3:
                    st.subheader("Performance by Phase")
                    
                    # Get phase data
                    phase_data = batter_profile.get('by_phase', {})
                    phase_names = {1: "Powerplay (1-6)", 2: "Middle Overs (7-16)", 3: "Death Overs (17-20)", 4: "Other"}
                    
                    if phase_data:
                        # Create phase selection dropdown
                        phase_options = ["All Phases"] + [f"{phase}: {phase_names.get(phase, f'Phase {phase}')}" 
                                                          for phase in sorted(phase_data.keys())]
                        selected_phase_option = st.selectbox("Select Phase", phase_options, key="phase_dropdown")
                        
                        # Process selection
                        if selected_phase_option == "All Phases":
                            # Prepare data for all phases
                            phases_df = []
                            for phase, stats in phase_data.items():
                                if stats['balls'] >= 5:  # Only include if sufficient data
                                    phases_df.append({
                                        'phase': phase_names.get(phase, f"Phase {phase}"),
                                        'strike_rate': stats['strike_rate'],
                                        'average': stats['average'] if stats['average'] != float('inf') else None,
                                        'runs': stats['runs'],
                                        'balls': stats['balls'],
                                        'dismissals': stats['dismissals']
                                    })
                            
                            if phases_df:
                                phases_df = pd.DataFrame(phases_df)
                                
                                # Display strike rate comparison
                                fig, ax = plt.subplots(figsize=(10, 6))
                                bars = ax.bar(phases_df['phase'], phases_df['strike_rate'], color='lightgreen')
                                
                                # Add data labels
                                for i, bar in enumerate(bars):
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                            f"{height:.1f}", ha='center', va='bottom', rotation=0)
                                
                                ax.set_title(f"{selected_batter}'s Strike Rate in Different Phases")
                                ax.set_xlabel("Phase")
                                ax.set_ylabel("Strike Rate")
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()
                                
                                st.pyplot(fig)
                                
                                # Display the data table
                                st.subheader("Phase Statistics")
                                st.dataframe(phases_df)
                            else:
                                st.write("Not enough data across phases for meaningful analysis.")
                        else:
                            # Extract the selected phase number
                            selected_phase = int(selected_phase_option.split(":")[0])
                            phase_stats = phase_data.get(selected_phase, {})
                            
                            if phase_stats:
                                st.write(f"### {selected_batter}'s Performance in {phase_names.get(selected_phase, f'Phase {selected_phase}')}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Strike Rate", f"{phase_stats['strike_rate']:.2f}")
                                with col2:
                                    avg_val = "No dismissals" if phase_stats['average'] == float('inf') else f"{phase_stats['average']:.2f}"
                                    st.metric("Average", avg_val)
                                with col3:
                                    st.metric("Runs / Balls", f"{phase_stats['runs']} / {phase_stats['balls']}")
                                
                                # Check if we have phase-bowl style data
                                if batter_profile.get('phase_bowl_style') and selected_phase in batter_profile['phase_bowl_style']:
                                    phase_bowlers = batter_profile['phase_bowl_style'][selected_phase]
                                    
                                    st.subheader(f"Performance Against Bowling Styles in {phase_names.get(selected_phase, f'Phase {selected_phase}')}")
                                    
                                    if phase_bowlers:
                                        # Prepare data for visualization
                                        phase_bowl_data = []
                                        for style, stats in phase_bowlers.items():
                                            phase_bowl_data.append({
                                                'style': style,
                                                'strike_rate': stats['strike_rate'],
                                                'average': stats['average'] if stats['average'] != float('inf') else None,
                                                'balls': stats['balls']
                                            })
                                        
                                        if phase_bowl_data:
                                            # Create DataFrame and sort by strike rate
                                            phase_bowl_df = pd.DataFrame(phase_bowl_data)
                                            phase_bowl_df = phase_bowl_df.sort_values('strike_rate', ascending=False)
                                            
                                            # Display bar chart
                                            fig, ax = plt.subplots(figsize=(12, 6))
                                            bars = ax.bar(phase_bowl_df['style'], phase_bowl_df['strike_rate'], color='orange')
                                            
                                            # Add data labels
                                            for i, bar in enumerate(bars):
                                                height = bar.get_height()
                                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                                        f"{height:.1f}", ha='center', va='bottom', rotation=0)
                                            
                                            ax.set_title(f"{selected_batter}'s Strike Rate vs Bowling Styles in {phase_names.get(selected_phase)}")
                                            ax.set_xlabel("Bowling Style")
                                            ax.set_ylabel("Strike Rate")
                                            plt.xticks(rotation=45, ha='right')
                                            plt.tight_layout()
                                            
                                            st.pyplot(fig)
                                            
                                            # Data table
                                            st.dataframe(phase_bowl_df)
                                    else:
                                        st.write("No detailed bowling style data available for this phase.")
                                else:
                                    st.write("No detailed bowling style data available for this phase.")
                            else:
                                st.write(f"No data available for {selected_batter} in {phase_names.get(selected_phase, f'Phase {selected_phase}')}.")
                    else:
                        st.write("No phase-wise data available for this batter.")
                
                with tab4:
                    st.subheader("Performance by Bowling Style")
                    
                    # Check if bowl_style column exists in the data
                    if 'bowl_style' not in st.session_state.data.columns:
                        st.warning("The dataset does not contain bowling style information. This tab requires the 'bowl_style' column.")
                    else:
                        # Get bowling styles faced by this batter
                        batter_data = st.session_state.data[st.session_state.data['bat'] == selected_batter]
                        batter_styles = batter_data['bowl_style'].dropna().unique()
                        valid_styles = [s for s in batter_styles if s != '-' and s != '']
                        
                        if not valid_styles:
                            st.info(f"No valid bowling style data found for {selected_batter}.")
                        else:
                            # Create a simple analysis of styles on the fly if needed
                            if not batter_profile.get('vs_bowler_styles'):
                                st.info("Building bowling style analysis directly from dataset...")
                                
                                # Create a dictionary to hold the style data
                                style_data = {}
                                
                                for style in valid_styles:
                                    style_balls = batter_data[batter_data['bowl_style'] == style]
                                    
                                    if len(style_balls) >= 3:  # Minimum 3 balls for analysis
                                        runs = style_balls['score'].sum()
                                        balls = len(style_balls)
                                        outs = style_balls['out'].sum()
                                        
                                        sr = (runs / balls * 100) if balls > 0 else 0
                                        avg = (runs / outs) if outs > 0 else float('inf')
                                        
                                        style_data[style] = {
                                            'style': style,
                                            'runs': int(runs),
                                            'balls': int(balls),
                                            'dismissals': int(outs),
                                            'strike_rate': sr,
                                            'average': avg
                                        }
                                
                                # Create a DataFrame for visualization
                                if style_data:
                                    styles_df = pd.DataFrame([v for v in style_data.values()])
                                    styles_df = styles_df.sort_values('strike_rate', ascending=False)
                                    
                                    # Display bar chart
                                    st.subheader(f"{selected_batter}'s Performance by Bowling Style")
                                    
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    bars = ax.bar(styles_df['style'], styles_df['strike_rate'], color='coral')
                                    
                                    # Add data labels
                                    for i, bar in enumerate(bars):
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                                f"{height:.1f}", ha='center', va='bottom', rotation=0)
                                    
                                    ax.set_title(f"{selected_batter}'s Strike Rate vs Different Bowling Styles")
                                    ax.set_xlabel("Bowling Style")
                                    ax.set_ylabel("Strike Rate")
                                    plt.xticks(rotation=45, ha='right')
                                    plt.tight_layout()
                                    
                                    st.pyplot(fig)
                                    
                                    # Display data table
                                    st.subheader("Style Statistics")
                                    st.dataframe(styles_df)
                                    
                                    # Add explanation of bowling styles
                                    style_explanations = {
                                        'RF': 'Right-arm Fast',
                                        'RFM': 'Right-arm Fast-Medium',
                                        'LB': 'Leg Break',
                                        'LWS': 'Left-arm Wrist Spin',
                                        'RMF': 'Right-arm Medium-Fast',
                                        'SLA': 'Slow Left-arm Orthodox',
                                        'OB': 'Off Break',
                                        'LBG': 'Leg Break Googly',
                                        'LFM': 'Left-arm Fast-Medium',
                                        'LF': 'Left-arm Fast',
                                        'RM': 'Right-arm Medium',
                                        'LMF': 'Left-arm Medium-Fast',
                                        'LM': 'Left-arm Medium'
                                    }
                                    
                                    st.subheader("Bowling Style Legend")
                                    legend_data = []
                                    for style in sorted(style_data.keys()):
                                        explanation = style_explanations.get(style, f"Unknown style ({style})")
                                        legend_data.append({"Style": style, "Description": explanation})
                                    
                                    st.table(pd.DataFrame(legend_data))
                                else:
                                    st.info(f"Not enough data for meaningful bowling style analysis for {selected_batter}.")
                            else:
                                # Use the pre-calculated profiles if available
                                bowl_styles = batter_profile.get('vs_bowler_styles', {})
                                
                                if bowl_styles:
                                    # Get all available bowling styles
                                    if hasattr(st.session_state.batter_analyzer, 'get_available_bowl_styles'):
                                        available_styles = st.session_state.batter_analyzer.get_available_bowl_styles()
                                    else:
                                        available_styles = sorted(bowl_styles.keys())
                                    
                                    # Create style selection dropdown
                                    style_options = ["All Styles"] + available_styles
                                    selected_style_option = st.selectbox("Select Bowling Style", style_options, key="style_dropdown")
                                    
                                    if selected_style_option == "All Styles":
                                        # Display data for all styles
                                        style_data = []
                                        for style, stats in bowl_styles.items():
                                            style_data.append({
                                                'style': style,
                                                'strike_rate': stats['strike_rate'],
                                                'average': stats['average'] if stats['average'] != float('inf') else None,
                                                'runs': stats['runs'],
                                                'balls': stats['balls'],
                                                'dismissals': stats['dismissals']
                                            })
                                        
                                        if style_data:
                                            # Convert to DataFrame and sort
                                            style_df = pd.DataFrame(style_data)
                                            style_df = style_df.sort_values('strike_rate', ascending=False)
                                            
                                            # Display bar chart for strike rates
                                            fig, ax = plt.subplots(figsize=(12, 8))
                                            bars = ax.bar(style_df['style'], style_df['strike_rate'], color='coral')
                                            
                                            # Add data labels
                                            for i, bar in enumerate(bars):
                                                height = bar.get_height()
                                                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                                        f"{height:.1f}", ha='center', va='bottom', rotation=0)
                                            
                                            ax.set_title(f"{selected_batter}'s Strike Rate vs Different Bowling Styles")
                                            ax.set_xlabel("Bowling Style")
                                            ax.set_ylabel("Strike Rate")
                                            plt.xticks(rotation=45, ha='right')
                                            plt.tight_layout()
                                            
                                            st.pyplot(fig)
                                            
                                            # Add tooltip explanation of bowling styles
                                            st.caption("""
                                            **Bowling Style Legend:**
                                            - RF: Right-arm Fast
                                            - RFM: Right-arm Fast-Medium
                                            - LB: Leg Break
                                            - LWS: Left-arm Wrist Spin
                                            - RMF: Right-arm Medium-Fast
                                            - SLA: Slow Left-arm Orthodox
                                            - OB: Off Break
                                            - LBG: Leg Break Googly
                                            - LFM: Left-arm Fast-Medium
                                            - LF: Left-arm Fast
                                            - RM: Right-arm Medium
                                            - LMF: Left-arm Medium-Fast
                                            """)
                                            
                                            # Display data table
                                            st.subheader("Style Statistics")
                                            st.dataframe(style_df)
                                        else:
                                            st.write("Not enough data across bowling styles for meaningful analysis.")
                                    else:
                                        # Show specific style analysis
                                        selected_style = selected_style_option
                                        style_stats = bowl_styles.get(selected_style)
                                        
                                        if style_stats:
                                            st.write(f"### {selected_batter}'s Performance Against {selected_style}")
                                            
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("Strike Rate", f"{style_stats['strike_rate']:.2f}")
                                            with col2:
                                                avg_val = "No dismissals" if style_stats['average'] == float('inf') else f"{style_stats['average']:.2f}"
                                                st.metric("Average", avg_val)
                                            with col3:
                                                st.metric("Runs / Balls", f"{style_stats['runs']} / {style_stats['balls']}")
                                            
                                            # Check if we have phase data for this style
                                            phase_style_data = []
                                            if batter_profile.get('phase_bowl_style'):
                                                for phase, styles in batter_profile['phase_bowl_style'].items():
                                                    if selected_style in styles:
                                                        phase_style_data.append({
                                                            'phase': phase_names.get(phase, f"Phase {phase}"),
                                                            'phase_num': phase,
                                                            'strike_rate': styles[selected_style]['strike_rate'],
                                                            'average': styles[selected_style]['average'] if styles[selected_style]['average'] != float('inf') else None,
                                                            'balls': styles[selected_style]['balls']
                                                        })
                                            
                                            if phase_style_data:
                                                st.subheader(f"Performance Against {selected_style} by Phase")
                                                
                                                # Create DataFrame and sort by phase
                                                phase_style_df = pd.DataFrame(phase_style_data)
                                                phase_style_df = phase_style_df.sort_values('phase_num')
                                                
                                                # Display bar chart
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                bars = ax.bar(phase_style_df['phase'], phase_style_df['strike_rate'], color='purple')
                                                
                                                # Add data labels
                                                for i, bar in enumerate(bars):
                                                    height = bar.get_height()
                                                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                                            f"{height:.1f}", ha='center', va='bottom', rotation=0)
                                                
                                                ax.set_title(f"{selected_batter}'s Strike Rate vs {selected_style} Across Phases")
                                                ax.set_xlabel("Phase")
                                                ax.set_ylabel("Strike Rate")
                                                plt.xticks(rotation=45, ha='right')
                                                plt.tight_layout()
                                                
                                                st.pyplot(fig)
                                                
                                                # Display data table
                                                st.dataframe(phase_style_df[['phase', 'strike_rate', 'average', 'balls']])
                                            else:
                                                st.write("No phase-specific data available for this bowling style.")
                                        else:
                                            st.write(f"No data available for {selected_batter} against {selected_style}.")
                                else:
                                    st.write("No detailed bowling style data available in the pre-processed profiles.")
elif st.session_state.current_page == "Bowler Strategies":
    st.header("Optimal Bowling Strategies")
    
    # Bowler selection
    bowlers = sorted(st.session_state.data['bowl'].unique())
    selected_bowler = st.selectbox("Select Bowler", bowlers)
    
    if selected_bowler:
        # Get bowler profile
        bowler_profile = st.session_state.bowler_analyzer.get_bowler_profile(selected_bowler)
        
        if bowler_profile:
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Bowler Profile")
                st.write(f"Bowling Style: {bowler_profile.get('bowl_style', 'Unknown')}")
                st.write(f"Bowler Type: {bowler_profile.get('bowl_kind', 'Unknown')}")
                st.write(f"Economy Rate: {bowler_profile.get('economy', 0):.2f}")
                
                # Handle inf average/strike rate
                avg = bowler_profile.get('average', 0)
                avg_display = f"{avg:.2f}" if avg != float('inf') else "No wickets"
                st.write(f"Average: {avg_display}")
                
                sr = bowler_profile.get('strike_rate', 0)
                sr_display = f"{sr:.2f}" if sr != float('inf') else "No wickets"
                st.write(f"Strike Rate: {sr_display}")
                
                st.write(f"Total Wickets: {bowler_profile.get('wickets', 0)}")
                st.write(f"Total Runs Conceded: {bowler_profile.get('total_runs', 0)}")
                st.write(f"Total Balls Bowled: {bowler_profile.get('total_balls', 0)}")
            
            with col2:
                st.subheader("Optimal Line & Length")
                
                # Get optimal line and length combinations
                optimal_combinations = st.session_state.bowler_analyzer.get_optimal_line_length(selected_bowler)
                
                if optimal_combinations:
                    for i, combo in enumerate(optimal_combinations, 1):
                        effectiveness = combo.get('effectiveness', 0)
                        economy = combo.get('economy', 0)
                        strike_rate = combo.get('strike_rate', float('inf'))
                        sr_display = f"{strike_rate:.2f}" if strike_rate != float('inf') else "No wickets"
                        
                        st.write(f"**{i}. {combo['line']} {combo['length']}**")
                        st.write(f"   Economy: {economy:.2f}")
                        st.write(f"   Strike Rate: {sr_display}")
                        st.write(f"   Effectiveness Score: {effectiveness:.2f}")
                        st.write(f"   Sample Size: {combo['sample_size']} balls")
                        
                        if i < len(optimal_combinations):
                            st.markdown("---")
                else:
                    st.write("Not enough data to recommend optimal line and length combinations.")
            
            # Display line and length effectiveness visualization
            st.subheader("Line & Length Effectiveness")
            
            line_length_stats = bowler_profile.get('by_line_length', {})
            if line_length_stats:
                # Create separate heatmaps for economy and strike rate
                tab1, tab2 = st.tabs(["Economy Rate", "Strike Rate"])
                
                with tab1:
                    economy_fig = create_bowler_economy_heatmap(line_length_stats)
                    st.pyplot(economy_fig)
                
                with tab2:
                    strike_rate_fig = create_bowler_strike_rate_heatmap(line_length_stats)
                    st.pyplot(strike_rate_fig)
            else:
                st.write("Not enough data to visualize line and length effectiveness.")

elif st.session_state.current_page == "Match-up Optimization":
    st.header("Batter-Bowler Match-up Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Batter selection
        batters = sorted(st.session_state.data['bat'].unique())
        selected_batter = st.selectbox("Select Batter", batters)
    
    with col2:
        # Bowler type selection (for filtering)
        all_bowl_types = st.session_state.data['bowl_kind'].unique()
        selected_types = st.multiselect(
            "Filter by Bowler Types (optional)",
            options=all_bowl_types,
            default=None
        )
    
    if selected_batter:
        st.subheader(f"Optimal Bowling Strategies vs {selected_batter}")
        
        # Get all available bowlers
        available_bowlers = []
        
        for bowler in st.session_state.data['bowl'].unique():
            bowler_profile = st.session_state.bowler_analyzer.get_bowler_profile(bowler)
            if bowler_profile:
                bowl_type = bowler_profile.get('bowl_kind', 'Unknown')
                
                # Apply filter if selected
                if not selected_types or bowl_type in selected_types:
                    available_bowlers.append({
                        'name': bowler,
                        'type': bowl_type,
                        'style': bowler_profile.get('bowl_style', 'Unknown')
                    })
        
        if hasattr(st.session_state, 'plan_generator') and st.session_state.plan_generator:
            # Get matchup recommendations
            matchups = st.session_state.plan_generator.identify_optimal_matchups(
                selected_batter, available_bowlers
            )
            
            if matchups and matchups.get('matchups'):
                # Display results in a table
                matchup_data = []
                
                for matchup in matchups['matchups']:
                    line_length_rec = ""
                    if matchup.get('recommendations') and len(matchup['recommendations']) > 0:
                        rec = matchup['recommendations'][0]
                        line_length_rec = f"{rec['line']} {rec['length']}"
                    
                    matchup_data.append({
                        'Bowler': matchup['bowler'],
                        'Type': matchup['type'],
                        'Matchup Score': f"{matchup['matchup_score']:.2f}",
                        'Recommended Line/Length': line_length_rec,
                        'Confidence': matchup['confidence'].upper(),
                        'Sample Size': matchup['sample_size']
                    })
                
                # Convert to DataFrame for display
                df = pd.DataFrame(matchup_data)
                
                # Style the dataframe
                st.dataframe(df.style.highlight_min(subset=['Matchup Score'], color='lightgreen'))
                
                # Show detailed analysis for top matchup
                if len(matchups['matchups']) > 0:
                    st.subheader(f"Detailed Strategy: {matchups['matchups'][0]['bowler']}")
                    
                    best_matchup = matchups['matchups'][0]
                    
                    # Get field settings
                    field_settings = {
                        'catching_positions': ['slip', 'gully', 'mid-off'],
                        'boundary_riders': ['deep cover', 'long-off', 'deep midwicket'],
                        'description': f"Recommended field for {best_matchup['bowler']} vs {selected_batter}"
                    }
                    
                    # Display field visualization
                    try:
                        field_fig = create_field_placement_visualization(field_settings)
                        st.pyplot(field_fig)
                    except Exception as e:
                        st.error(f"Error creating field visualization: {e}")
                
            else:
                st.write("Not enough data to generate matchup recommendations.")
        else:
            st.write("Plan generator not initialized. Cannot analyze matchups.")

elif st.session_state.current_page == "Complete Bowling Plan":
    st.header("Comprehensive Bowling Plan Generator")
    
    # Batter selection
    batters = sorted(st.session_state.data['bat'].unique())
    selected_batter = st.selectbox("Select Batter", batters)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Game phase emphasis
        phase_emphasis = st.select_slider(
            "Phase Emphasis",
            options=["Powerplay", "Balanced", "Death Overs"],
            value="Balanced"
        )
    
    with col2:
        # Strategy emphasis
        strategy_emphasis = st.select_slider(
            "Strategy Emphasis",
            options=["Economy", "Balanced", "Wicket-taking"],
            value="Balanced"
        )
    
    if selected_batter and hasattr(st.session_state, 'plan_generator') and st.session_state.plan_generator:
        # Get all available bowlers
        available_bowlers = []
        
        for bowler in st.session_state.data['bowl'].unique():
            bowler_profile = st.session_state.bowler_analyzer.get_bowler_profile(bowler)
            if bowler_profile:
                available_bowlers.append({
                    'name': bowler,
                    'type': bowler_profile.get('bowl_kind', 'Unknown'),
                    'style': bowler_profile.get('bowl_style', 'Unknown')
                })
        
        # Generate comprehensive bowling plan
        plan = st.session_state.plan_generator.generate_complete_bowling_plan(
            selected_batter, available_bowlers
        )
        
        if plan and not plan.get('error'):
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Summary", "Phase-by-Phase Plan", "Matchups"])
            
            with tab1:
                # Display summary visualization
                try:
                    summary_fig = get_strategy_visualization(plan)
                    st.pyplot(summary_fig)
                except Exception as e:
                    st.error(f"Error creating summary visualization: {e}")
                
                # Display text summary
                st.subheader("Strategy Summary")
                st.text(plan.get('summary', 'No summary available.'))
            
            with tab2:
                # Display phase-specific plans
                phase_plans = plan.get('phase_plans', {})
                
                if phase_plans:
                    for phase_id, phase_plan in phase_plans.items():
                        phase_name = phase_names.get(int(phase_id), f"Phase {phase_id}")
                        
                        with st.expander(phase_name, expanded=True):
                            st.write(f"**Optimal Bowler Type:** {phase_plan.get('optimal_bowler_type', 'Unknown')}")
                            
                            # Line and length recommendations
                            recs = phase_plan.get('line_length_recommendations', [])
                            if recs:
                                st.write("**Bowling Recommendations:**")
                                for i, rec in enumerate(recs, 1):
                                    st.write(f"{i}. {rec.get('line', '')} {rec.get('length', '')}")
                                    if 'batter_strike_rate' in rec:
                                        st.write(f"   Batter Strike Rate: {rec['batter_strike_rate']:.2f}")
                                    if 'bowler_economy' in rec:
                                        st.write(f"   Bowler Economy: {rec['bowler_economy']:.2f}")
                            
                            # Field setting
                            field_setting = phase_plan.get('field_setting', {})
                            if field_setting:
                                st.write(f"**Field Setting:** {field_setting.get('description', '')}")
                                
                                # Show field visualization if available
                                try:
                                    field_fig = create_field_placement_visualization(field_setting)
                                    st.pyplot(field_fig)
                                except Exception as e:
                                    st.write("Could not create field visualization.")
                else:
                    st.write("No phase-specific plans available.")
            
            with tab3:
                # Display optimal matchups
                matchups = plan.get('optimal_matchups', [])
                
                if matchups:
                    st.subheader("Optimal Bowler Matchups")
                    
                    matchup_data = []
                    for matchup in matchups[:5]:  # Show top 5
                        matchup_data.append({
                            'Bowler': matchup['bowler'],
                            'Type': matchup['type'],
                            'Matchup Score': f"{matchup['matchup_score']:.2f}" 
                        })
                    
                    # Convert to DataFrame for display
                    df = pd.DataFrame(matchup_data)
                    st.table(df)
                else:
                    st.write("No matchup data available.")
            
            # Export options
            st.subheader("Export Plan")
            
            # Create a function to export the plan as HTML
            def export_plan_as_html():
                html_content = f"""
                <html>
                <head>
                    <title>Bowling Plan for {selected_batter}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2, h3 {{ color: #1e4e8c; }}
                        .section {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                        .phase {{ background-color: #f5f5f5; padding: 10px; margin-bottom: 10px; }}
                        .recommendation {{ margin-left: 20px; }}
                    </style>
                </head>
                <body>
                    <h1>T20 Bowling Strategy: {selected_batter}</h1>
                    
                    <div class="section">
                        <h2>Batter Profile</h2>
                        <p>Batting Hand: {plan.get('batter_info', {}).get('bat_hand', 'Unknown')}</p>
                        <p>Strike Rate: {plan.get('batter_info', {}).get('strike_rate', 0):.2f}</p>
                        <p>Average: {plan.get('batter_info', {}).get('average', 0):.2f}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Key Weaknesses</h2>
                """
                
                # Add weaknesses
                if plan.get('weaknesses'):
                    for weakness in plan['weaknesses']:
                        if weakness['type'] == 'bowler_type':
                            html_content += f"<p>• Vulnerable against {weakness['bowler_type']}</p>"
                        elif weakness['type'] == 'line_length':
                            html_content += f"<p>• Struggles against {weakness['line']} {weakness['length']} deliveries</p>"
                        elif weakness['type'] == 'phase':
                            phase = phase_names.get(weakness['phase'], weakness['phase'])
                            html_content += f"<p>• Less effective in {phase}</p>"
                else:
                    html_content += "<p>No significant weaknesses identified.</p>"
                
                html_content += """
                    </div>
                    
                    <div class="section">
                        <h2>Phase-by-Phase Strategy</h2>
                """
                
                # Add phase plans
                for phase_id, phase_plan in plan.get('phase_plans', {}).items():
                    phase_name = phase_names.get(int(phase_id), f"Phase {phase_id}")
                    
                    html_content += f"""
                        <div class="phase">
                            <h3>{phase_name}</h3>
                            <p><strong>Optimal Bowler Type:</strong> {phase_plan.get('optimal_bowler_type', 'Unknown')}</p>
                            <p><strong>Bowling Recommendations:</strong></p>
                    """
                    
                    # Line and length recommendations
                    recs = phase_plan.get('line_length_recommendations', [])
                    if recs:
                        for i, rec in enumerate(recs, 1):
                            html_content += f"""
                                <div class="recommendation">
                                    <p>{i}. {rec.get('line', '')} {rec.get('length', '')}</p>
                            """
                            
                            if 'batter_strike_rate' in rec:
                                html_content += f"<p>   Batter Strike Rate: {rec['batter_strike_rate']:.2f}</p>"
                            if 'bowler_economy' in rec:
                                html_content += f"<p>   Bowler Economy: {rec['bowler_economy']:.2f}</p>"
                            
                            html_content += "</div>"
                    
                    # Field setting
                    field_setting = phase_plan.get('field_setting', {})
                    if field_setting:
                        html_content += f"""
                            <p><strong>Field Setting:</strong> {field_setting.get('description', '')}</p>
                            <ul>
                        """
                        
                        for pos in field_setting.get('catching_positions', []):
                            html_content += f"<li>Catching: {pos}</li>"
                        
                        for pos in field_setting.get('boundary_riders', []):
                            html_content += f"<li>Boundary: {pos}</li>"
                        
                        html_content += "</ul>"
                    
                    html_content += "</div>"
                
                html_content += """
                    </div>
                    
                    <div class="section">
                        <h2>Optimal Bowler Matchups</h2>
                        <table border="1" cellpadding="5" style="border-collapse: collapse; width: 100%;">
                            <tr style="background-color: #1e4e8c; color: white;">
                                <th>Bowler</th>
                                <th>Type</th>
                                <th>Matchup Score</th>
                            </tr>
                """
                
                # Add matchups
                matchups = plan.get('optimal_matchups', [])
                for matchup in matchups[:5]:  # Show top 5
                    html_content += f"""
                        <tr>
                            <td>{matchup['bowler']}</td>
                            <td>{matchup['type']}</td>
                            <td>{matchup['matchup_score']:.2f}</td>
                        </tr>
                    """
                
                html_content += """
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Summary</h2>
                        <pre style="white-space: pre-wrap;">{}</pre>
                    </div>
                    
                    <footer style="margin-top: 30px; text-align: center; color: #666;">
                        Generated by T20 Bowling Strategy Analyzer
                    </footer>
                </body>
                </html>
                """.format(plan.get('summary', 'No summary available.'))
                
                return html_content
            
            # Create export buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Export as HTML
                html_export = export_plan_as_html()
                st.download_button(
                    "Export as HTML",
                    html_export,
                    f"bowling_plan_{selected_batter.replace(' ', '_')}.html",
                    "text/html",
                    key="export-html"
                )
            
            with col2:
                # Export as plain text
                text_export = plan.get('summary', 'No summary available.')
                st.download_button(
                    "Export as Text",
                    text_export,
                    f"bowling_plan_{selected_batter.replace(' ', '_')}.txt",
                    "text/plain",
                    key="export-text"
                )
        else:
            st.error(f"Could not generate bowling plan: {plan.get('error', 'Unknown error')}")
            st.write("This may be due to insufficient data for the selected batter.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit • [GitHub Repository](https://github.com/prahaladvathsan/t20-bowling-strategy-analyzer)")