"""Common metric display components"""
import streamlit as st
import pandas as pd

def display_batter_metrics(batter_profile):
    """Display key metrics for a batter"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Strike Rate", f"{batter_profile['strike_rate']:.2f}")
        if batter_profile.get('effective_strike_rate'):
            st.metric("Effective SR", f"{batter_profile['effective_strike_rate']:.2f}")
    
    with col2:
        st.metric("Average", f"{batter_profile['average']:.2f}")
        if batter_profile.get('effective_average'):
            st.metric("Effective Avg", f"{batter_profile['effective_average']:.2f}")
    
    with col3:
        st.metric("Dot Ball %", f"{(batter_profile['dot_balls'] / batter_profile['total_balls'] * 100):.1f}%")
        st.metric("Dismissal Rate", f"{(batter_profile['dismissals'] / batter_profile['total_balls'] * 100):.1f}%")

def display_bowler_metrics(bowler_stats):
    """Display common bowler metrics
    
    Parameters:
        bowler_stats (dict): Dictionary containing bowler statistics
    """
    st.write(f"Bowling Style: {bowler_stats.get('bowl_style', 'Unknown')}")
    st.write(f"Bowler Type: {bowler_stats.get('bowl_kind', 'Unknown')}")
    st.write(f"Economy Rate: {bowler_stats.get('economy', 0):.2f}")
    
    # Handle inf average/strike rate
    avg = bowler_stats.get('average', 0)
    avg_display = f"{avg:.2f}" if avg != float('inf') else "No wickets"
    st.write(f"Average: {avg_display}")
    
    sr = bowler_stats.get('strike_rate', 0)
    sr_display = f"{sr:.2f}" if sr != float('inf') else "No wickets"
    st.write(f"Strike Rate: {sr_display}")
    
    st.write(f"Total Wickets: {bowler_stats.get('wickets', 0)}")
    st.write(f"Total Runs Conceded: {bowler_stats.get('total_runs', 0)}")
    st.write(f"Total Balls Bowled: {bowler_stats.get('total_balls', 0)}")

def display_phase_metrics(phase_stats):
    """Display phase-specific metrics
    
    Parameters:
        phase_stats (dict): Dictionary containing phase statistics
    """
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Strike Rate", f"{phase_stats['strike_rate']:.2f}")
    with col2:
        st.metric("Average", f"{phase_stats['average']:.2f}")
    with col3:
        if 'effective_strike_rate' in phase_stats:
            st.metric("Effective SR ❓", f"{phase_stats['effective_strike_rate']:.2f}")
            st.caption("SR relative to team's SR in this phase")
    with col4:
        if 'effective_average' in phase_stats:
            st.metric("Effective Avg ❓", f"{phase_stats['effective_average']:.2f}")
            st.caption("Avg relative to team's avg in this phase")
    with col5:
        st.metric("Runs/Balls", f"{phase_stats['runs']}/{phase_stats['balls']}")

def display_weakness_summary(vulnerability_rankings):
    """Display summary of batter's key weaknesses"""
    if not vulnerability_rankings:
        st.info("No vulnerability data available")
        return
        
    st.subheader("Key Weaknesses")
    
    # Show bowling style vulnerabilities
    if 'vs_bowler_styles' in vulnerability_rankings:
        styles = vulnerability_rankings['vs_bowler_styles']
        if isinstance(styles, dict):
            # Sort styles by vulnerability score
            sorted_styles = sorted(
                [(style, data['vulnerability']) 
                 for style, data in styles.items()],
                key=lambda x: x[1],
                reverse=True
            )
            if sorted_styles:
                most_vulnerable_style = sorted_styles[0]
                st.write(f"🎯 Most vulnerable against: **{most_vulnerable_style[0]}**")
                st.write(f"Vulnerability score: {most_vulnerable_style[1]:.1f}")
    
    # Show phase vulnerabilities
    if 'by_phase' in vulnerability_rankings:
        phases = vulnerability_rankings['by_phase']
        if isinstance(phases, dict):
            # Sort phases by vulnerability score
            sorted_phases = sorted(
                [(phase, data['vulnerability']) 
                 for phase, data in phases.items()],
                key=lambda x: x[1],
                reverse=True
            )
            if sorted_phases:
                weakest_phase = sorted_phases[0]
                phase_map = {1: "Powerplay", 2: "Middle", 3: "Death"}
                phase_name = phase_map.get(weakest_phase[0], weakest_phase[0])
                st.write(f"⏰ Weakest phase: **{phase_name}**")
                st.write(f"Vulnerability score: {weakest_phase[1]:.1f}")
    
    # Show line-length vulnerabilities
    if 'vs_line_length' in vulnerability_rankings:
        ll_stats = vulnerability_rankings['vs_line_length']
        if isinstance(ll_stats, dict):
            # Sort line-length combinations by vulnerability score
            sorted_ll = sorted(
                [(f"{ll}", data['vulnerability']) 
                 for ll, data in ll_stats.items()],
                key=lambda x: x[1],
                reverse=True
            )
            if sorted_ll:
                weakest_ll = sorted_ll[0]
                st.write(f"📍 Struggles against: **{weakest_ll[0]}**")
                st.write(f"Vulnerability score: {weakest_ll[1]:.1f}")