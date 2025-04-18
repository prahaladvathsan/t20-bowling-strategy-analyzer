import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Import modules
from data_processor import DataProcessor
from batter_analyzer import BatterVulnerabilityAnalyzer
from bowler_analyzer import BowlerAnalyzer
from bowling_plan_generator import BowlingPlanGenerator
from utils.visualization import (
    create_vulnerability_heatmap,
    create_field_placement_visualization,
    get_strategy_visualization,
    create_bowler_economy_heatmap,
    create_bowler_strike_rate_heatmap,
    create_phase_vulnerability_heatmap,
    create_style_vulnerability_heatmap
)

def load_analyzers():
    """Load analyzers with saved data"""
    try:
        # Initialize analyzers to load from saved files
        batter_analyzer = BatterVulnerabilityAnalyzer()
        bowler_analyzer = BowlerAnalyzer()
        plan_generator = BowlingPlanGenerator()
        
        return {
            'batter_analyzer': batter_analyzer,
            'bowler_analyzer': bowler_analyzer,
            'plan_generator': plan_generator
        }, None
    except Exception as e:
        return None, str(e)

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
    st.session_state.batter_analyzer = None
    st.session_state.bowler_analyzer = None
    st.session_state.plan_generator = None
    st.session_state.current_page = "Batter Analysis"
    st.session_state.app_started = False
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

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    
    if not st.session_state.analyzers_initialized:
        with st.spinner("Initializing analyzers..."):
            analyzers, error = load_analyzers()
            if error:
                st.error(f"Error initializing analyzers: {error}")
            else:
                st.session_state.batter_analyzer = analyzers['batter_analyzer']
                st.session_state.bowler_analyzer = analyzers['bowler_analyzer']
                st.session_state.plan_generator = analyzers['plan_generator']
                st.session_state.analyzers_initialized = True
                st.success("Analyzers initialized successfully")
    
    # Navigation options
    st.session_state.current_page = st.radio(
        "Select Analysis",
        ["Batter Analysis", "Bowler Strategies", "Match-up Optimization", "Complete Bowling Plan"],
        index=0,
        key="page_selection"
    )

# Main content area based on selection
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
    batters = sorted(st.session_state.batter_analyzer.get_all_batters())
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
                    
                    # Add effective metrics with explanatory tooltips
                    eff_sr = batter_profile.get('effective_strike_rate')
                    if eff_sr is not None:
                        st.write(f"Effective Strike Rate: {eff_sr:.2f} ❓")
                        st.caption("Effective SR = Batter's SR relative to team's SR while at crease")
                    
                    eff_avg = batter_profile.get('effective_average')
                    if eff_avg is not None:
                        st.write(f"Effective Average: {eff_avg:.2f} ❓")
                        st.caption("Effective Avg = Batter's avg relative to team's avg while at crease")
                    
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
                    
                    # Get bowling styles data
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
                    
                    # Phase data
                    phase_data = batter_profile.get('by_phase', {})
                    phase_line_length_data = batter_profile.get('phase_line_length', {})
                    
                    # Add phase selector
                    phase_options = {
                        '1': "Powerplay (Overs 1-6)",
                        '2': "Early Middle (Overs 7-12)",
                        '3': "Late Middle (Overs 13-16)",
                        '4': "Death (Overs 17-20)"
                    }
                    selected_phase = st.selectbox(
                        "Select Phase",
                        options=list(phase_options.keys()),
                        format_func=lambda x: phase_options[x],
                        key="phase_selector"
                    )
                    
                    if selected_phase:
                        # Display phase performance metrics in columns
                        if phase_data and selected_phase in phase_data:
                            phase_stats = phase_data[selected_phase]
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Strike Rate", f"{phase_stats['strike_rate']:.2f}")
                            with col2:
                                st.metric("Average", f"{phase_stats['average']:.2f}")
                            with col3:
                                # Add effective strike rate if available
                                if 'effective_strike_rate' in phase_stats:
                                    st.metric("Effective SR ❓", f"{phase_stats['effective_strike_rate']:.2f}")
                                    st.caption("SR relative to team's SR in this phase")
                            with col4:
                                # Add effective average if available
                                if 'effective_average' in phase_stats:
                                    st.metric("Effective Avg ❓", f"{phase_stats['effective_average']:.2f}")
                                    st.caption("Avg relative to team's avg in this phase")
                            with col5:
                                st.metric("Runs/Balls", f"{phase_stats['runs']}/{phase_stats['balls']}")
                        else:
                            st.write("No data available for this phase.")
                        
                        # Display phase-wise line-length heatmap below the stats
                        if phase_line_length_data and selected_phase in phase_line_length_data:
                            phase_ll_stats = phase_line_length_data[selected_phase]
                            try:
                                st.write("") # Add some spacing
                                heatmap_fig = create_phase_vulnerability_heatmap(phase_ll_stats)
                                st.pyplot(heatmap_fig)
                            except Exception as e:
                                st.error(f"Error creating phase vulnerability heatmap: {e}")
                                st.write("Not enough data to create a meaningful heatmap for this phase.")
                        else:
                            st.write("Not enough line-length data available for this phase.")
                
                with tab4:
                    st.subheader("Performance by Bowling Style")
                    
                    # Get bowling style data
                    bowl_styles = st.session_state.batter_analyzer.get_available_bowl_styles()
                    
                    # Add bowling style selector
                    selected_style = st.selectbox(
                        "Select Bowling Style",
                        options=bowl_styles,
                        key="style_selector"
                    )
                    
                    if selected_style:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Display style performance metrics
                            style_stats = st.session_state.batter_analyzer.analyze_batter_vs_bowl_style(selected_batter, selected_style)
                            if style_stats:
                                # Standard metrics
                                st.metric("Strike Rate", style_stats['strike_rate'])
                                st.metric("Average", (style_stats['average']))
                                
                                # Add effective metrics with explanatory tooltips
                                if 'effective_strike_rate' in style_stats:
                                    st.metric("Effective SR ❓", style_stats['effective_strike_rate'])
                                    st.caption("SR relative to team's SR vs this bowling style")
                                    
                                if 'effective_average' in style_stats:
                                    st.metric("Effective Avg ❓", style_stats['effective_average'])
                                    st.caption("Avg relative to team's avg vs this bowling style")
                                    
                                st.metric("Runs/Balls", f"{style_stats['runs']}/{style_stats['balls']}")
                            else:
                                st.write("No data available for this bowling style.")
                        
                        with col2:
                            # Display style-wise line-length heatmap
                            style_ll_stats = st.session_state.batter_analyzer.analyze_style_line_length(selected_batter, selected_style)
                            if style_ll_stats:
                                try:
                                    heatmap_fig = create_style_vulnerability_heatmap(style_ll_stats)
                                    st.pyplot(heatmap_fig)
                                except Exception as e:
                                    st.error(f"Error creating style vulnerability heatmap: {e}")
                                    st.write("Not enough data to create a meaningful heatmap for this bowling style.")
                            else:
                                st.write("Not enough line-length data available for this bowling style.")
elif st.session_state.current_page == "Bowler Strategies":
    st.header("Optimal Bowling Strategies")
    
    # Bowler selection
    bowlers = sorted(st.session_state.bowler_analyzer.get_all_bowlers())
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
        batters = sorted(st.session_state.batter_analyzer.get_all_batters())
        selected_batter = st.selectbox("Select Batter", batters)
    
    with col2:
        # Bowler type selection (for filtering)
        all_bowl_types = st.session_state.bowler_analyzer.get_all_bowler_types()
        selected_types = st.multiselect(
            "Filter by Bowler Types (optional)",
            options=all_bowl_types,
            default=None
        )
    
    if selected_batter:
        st.subheader(f"Optimal Bowling Strategies vs {selected_batter}")
        
        # Get all available bowlers
        available_bowlers = st.session_state.bowler_analyzer.get_filtered_bowlers(selected_types)
        
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
    batters = sorted(st.session_state.batter_analyzer.get_all_batters())
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
        available_bowlers = st.session_state.bowler_analyzer.get_all_bowlers()
        
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