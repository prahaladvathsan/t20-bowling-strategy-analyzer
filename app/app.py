import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules
from src.data_processor import DataProcessor
from src.batter_analyzer import BatterVulnerabilityAnalyzer
from src.bowler_analyzer import BowlerAnalyzer 
from src.bowling_plan_generator import BowlingPlanGenerator
from app.utils.visualization import (
    create_vulnerability_heatmap,
    create_field_placement_visualization,
    get_strategy_visualization
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

# App title and description
st.title("🏏 T20 Bowling Strategy Analyzer")
st.markdown("""
This application helps cricket teams develop data-driven bowling strategies 
for T20 matches based on ball-by-ball analysis.
""")

# Sidebar for data upload and global controls
with st.sidebar:
    st.header("Data Input")
    
    # Option to use sample data
    use_sample = st.checkbox("Use sample data")
    
    if use_sample:
        # Load sample data (you'll need to include this in your repo)
        sample_path = Path(__file__).parent.parent / "data" / "sample_t20_data.csv"
        if sample_path.exists():
            data = pd.read_csv(sample_path)
            st.success(f"Loaded sample data with {len(data)} records")
            
            # Process the data
            data_processor = DataProcessor(data)
            processed_data = data_processor.process()
            
            # Store in session state
            st.session_state.data = processed_data
            st.session_state.batter_analyzer = BatterVulnerabilityAnalyzer(processed_data)
            st.session_state.bowler_analyzer = BowlerAnalyzer(processed_data)
            
            # Initialize plan generator if not already done
            if not hasattr(st.session_state, 'plan_generator') or st.session_state.plan_generator is None:
                st.session_state.plan_generator = BowlingPlanGenerator(processed_data)
        else:
            st.error("Sample data file not found. Please upload your own data.")
            use_sample = False
    
    if not use_sample:
        uploaded_file = st.file_uploader("Upload ball-by-ball data (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            # Load and process data
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(data)} records")
                
                # Process the data
                data_processor = DataProcessor(data)
                processed_data = data_processor.process()
                
                # Store in session state
                st.session_state.data = processed_data
                st.session_state.batter_analyzer = BatterVulnerabilityAnalyzer(processed_data)
                st.session_state.bowler_analyzer = BowlerAnalyzer(processed_data)
                
                # Initialize plan generator
                st.session_state.plan_generator = BowlingPlanGenerator(processed_data)
                
                # Display data stats
                st.subheader("Dataset Statistics")
                st.write(f"Matches: {processed_data['p_match'].nunique()}")
                st.write(f"Batters: {processed_data['bat'].nunique()}")
                st.write(f"Bowlers: {processed_data['bowl'].nunique()}")
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    st.markdown("---")
    st.header("Navigation")
    
    # Navigation options
    page = st.radio(
        "Select Analysis",
        ["Batter Analysis", "Bowler Strategies", "Match-up Optimization", "Complete Bowling Plan"]
    )

# Main content area based on selection
if st.session_state.data is None:
    # Display instructions if no data uploaded
    st.info("Please upload your ball-by-ball data CSV file or use sample data to begin analysis.")
    
    with st.expander("Sample Data Format"):
        st.markdown("""
        Your CSV should include the following columns:
        - `p_match`: Match ID
        - `bat`: Batter name
        - `bowl`: Bowler name
        - `line`: Bowling line (Off, Middle, Leg)
        - `length`: Bowling length (Yorker, Full, Good, Short)
        - `phase`: Game phase (1, 2, 3)
        - `score`: Runs scored on each ball
        - `out`: Boolean indicating dismissal
        
        Additional columns like `bat_hand`, `bowl_style`, and `bowl_kind` improve analysis quality.
        """)
    
    # Display sample CSV template for download
    sample_data = {
        'p_match': [1001, 1001, 1001, 1001, 1001],
        'inns': [1, 1, 1, 1, 1],
        'bat': ['Rohit Sharma', 'Rohit Sharma', 'Rohit Sharma', 'Virat Kohli', 'Virat Kohli'],
        'team_bat': ['India', 'India', 'India', 'India', 'India'], 
        'bowl': ['Shaheen Afridi', 'Shaheen Afridi', 'Haris Rauf', 'Haris Rauf', 'Shadab Khan'],
        'team_bowl': ['Pakistan', 'Pakistan', 'Pakistan', 'Pakistan', 'Pakistan'],
        'ball': [1, 2, 3, 4, 5],
        'score': [0, 4, 1, 2, 0],
        'out': [False, False, False, False, True],
        'phase': [1, 1, 1, 1, 1],
        'line': ['Off', 'Middle', 'Off', 'Leg', 'Middle'],
        'length': ['Good', 'Full', 'Short', 'Good', 'Good'],
        'bat_hand': ['RHB', 'RHB', 'RHB', 'RHB', 'RHB'],
        'bowl_kind': ['pace bowler', 'pace bowler', 'pace bowler', 'pace bowler', 'spin bowler']
    }
    
    sample_df = pd.DataFrame(sample_data)
    csv = sample_df.to_csv(index=False)
    
    st.download_button(
        "Download Sample CSV Template",
        csv,
        "sample_template.csv",
        "text/csv",
        key='download-sample'
    )
        
else:
    # Render appropriate page based on selection
    if page == "Batter Analysis":
        st.header("Batter Vulnerability Analysis")
        
        # Batter selection
        batters = sorted(st.session_state.data['bat'].unique())
        selected_batter = st.selectbox("Select Batter", batters)
        
        if selected_batter:
            # Get batter analysis
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
                                    phase_names = {1: 'Powerplay', 2: 'Middle Overs', 3: 'Death Overs'}
                                    phase = phase_names.get(weakness['phase'], weakness['phase'])
                                    st.write(f"👉 Less effective in {phase}")
                                    st.write(f"   Strike Rate: {weakness['strike_rate']:.2f} (Overall: {weakness['overall_strike_rate']:.2f})")
                                    st.write(f"   Confidence: {weakness['confidence'].upper()}")
                        else:
                            st.write("No significant weaknesses identified in the dataset.")
                    else:
                        st.write("Plan generator not initialized. Cannot analyze weaknesses.")
                
                # Display vulnerability heatmap
                st.subheader("Vulnerability Analysis")
                
                tab1, tab2 = st.tabs(["Vulnerability Heatmap", "Performance by Bowler Type"])
                
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
    
    elif page == "Bowler Strategies":
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
                    # Create data for heatmap
                    lines = ['Off', 'Middle', 'Leg']
                    lengths = ['Yorker', 'Full', 'Good', 'Short']
                    
                    # Create separate heatmaps for economy and strike rate
                    tab1, tab2 = st.tabs(["Economy Rate", "Strike Rate"])
                    
                    with tab1:
                        # Economy rate heatmap (lower is better)
                        economy_data = np.ones((len(lengths), len(lines))) * 10  # Default high value
                        
                        for (line, length), stats in line_length_stats.items():
                            if line in lines and length in lengths and stats['balls'] >= 5:
                                row_idx = lengths.index(length)
                                col_idx = lines.index(line)
                                economy_data[row_idx, col_idx] = stats['economy']
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(
                            economy_data,
                            annot=True,
                            fmt=".2f",
                            cmap="YlGnBu_r",  # Reverse colormap so blue = good (low economy)
                            xticklabels=lines,
                            yticklabels=lengths,
                            ax=ax,
                            vmin=min(6, economy_data.min()),  # Reasonable minimum
                            vmax=max(12, economy_data.max())  # Reasonable maximum
                        )
                        
                        ax.set_title(f"Economy Rate by Line & Length", fontsize=14)
                        ax.set_xlabel("Line", fontsize=12)
                        ax.set_ylabel("Length", fontsize=12)
                        
                        st.pyplot(fig)
                    
                    with tab2:
                        # Strike rate heatmap (lower is better)
                        strike_data = np.ones((len(lengths), len(lines))) * 50  # Default high value
                        
                        for (line, length), stats in line_length_stats.items():
                            if line in lines and length in lengths and stats['wickets'] > 0:
                                row_idx = lengths.index(length)
                                col_idx = lines.index(line)
                                strike_data[row_idx, col_idx] = stats['strike_rate']
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(
                            strike_data,
                            annot=True,
                            fmt=".1f",
                            cmap="YlGnBu_r",  # Reverse colormap so blue = good (low strike rate)
                            xticklabels=lines,
                            yticklabels=lengths,
                            ax=ax,
                            vmin=min(10, strike_data.min()),  # Reasonable minimum
                            vmax=min(50, strike_data.max())  # Reasonable maximum
                        )
                        
                        ax.set_title(f"Strike Rate by Line & Length", fontsize=14)
                        ax.set_xlabel("Line", fontsize=12)
                        ax.set_ylabel("Length", fontsize=12)
                        
                        st.pyplot(fig)
                else:
                    st.write("Not enough data to visualize line and length effectiveness.")
    
    elif page == "Match-up Optimization":
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
    
    elif page == "Complete Bowling Plan":
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
                        phase_names = {
                            1: "Powerplay (Overs 1-6)",
                            2: "Middle Overs (Overs 7-15)",
                            3: "Death Overs (Overs 16-20)"
                        }
                        
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
                                phase_names = {1: 'Powerplay', 2: 'Middle Overs', 3: 'Death Overs'}
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
                    phase_names = {
                        1: "Powerplay (Overs 1-6)",
                        2: "Middle Overs (Overs 7-15)",
                        3: "Death Overs (Overs 16-20)"
                    }
                    
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
st.markdown("Created with ❤️ using Streamlit • [GitHub Repository](https://github.com/yourusername/t20-bowling-strategy-analyzer)")