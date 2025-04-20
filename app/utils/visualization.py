import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import sys
from pathlib import Path

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from data_processor import DataProcessor

def create_common_heatmap(data, row_labels, col_labels, title, cmap='YlOrRd', value_label='Value', ball_counts=None):
    """Common function to create heatmaps with consistent styling"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a copy of data for visualization, replacing inf with nan to avoid display issues
    display_data = np.copy(data)
    display_data[np.isinf(display_data)] = np.nan
    
    # Create heatmap using imshow with fixed range for vulnerability heatmaps
    if 'Vulnerability' in title:
        im = ax.imshow(display_data, cmap=cmap, vmin=0, vmax=100)
    else:
        im = ax.imshow(display_data, cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(value_label, rotation=-90, va="bottom")
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = data[i, j]
            if np.isinf(value):
                text = '∞'  # Using infinity symbol
            else:
                text = f'{value:.2f}'
                
            if ball_counts is not None:
                text += f'\n({int(ball_counts[i, j])})'
            
            # Always use black text for better visibility
            text_color = "black"
            ax.text(j, i, text,
                ha="center", va="center",
                color=text_color,
                fontweight='bold',
                fontsize=8)

    # Configure axis labels and ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Line", fontsize=12)
    ax.set_ylabel("Length", fontsize=12)
    plt.tight_layout()
    
    return fig

def process_line_length_data(stats_dict, data_type='vulnerability'):
    """Process line/length data for heatmap creation"""
    # Create arrays of display names in order
    lines = [DataProcessor.LINE_DISPLAY[i] for i in range(5)]
    lengths = [DataProcessor.LENGTH_DISPLAY[i] for i in range(6)]
    
    data = np.zeros((6, 5))
    ball_counts = np.zeros((6, 5))
    
    if stats_dict:
        for combo_key, stats in stats_dict.items():
            try:
                # Parse the tuple of display names
                # Remove outer parentheses and split
                combo_str = combo_key.strip("()")
                line_display, length_display = [s.strip().strip("'") for s in combo_str.split(',')]
                
                # Find indices in our ordered arrays
                col_idx = lines.index(line_display)
                row_idx = lengths.index(length_display)
                
                if stats['balls'] >= 1:
                    if data_type == 'vulnerability':
                        value = stats['vulnerability']
                    elif data_type == 'economy':
                        value = stats['economy']
                    elif data_type == 'strike_rate':
                        value = stats['bowling_strike_rate'] if stats.get('wickets', 0) > 0 else np.inf
                    
                    data[row_idx, col_idx] = value
                    ball_counts[row_idx, col_idx] = stats['balls']
            except (ValueError, IndexError) as e:
                print(f"Error processing {combo_key}: {e}")
                continue
    
    return data, ball_counts, lines, lengths

def create_vulnerability_heatmap(batter_data):
    """Create a heatmap showing a batter's vulnerability to different line/length combinations"""
    data, ball_counts, lines, lengths = process_line_length_data(
        batter_data.get('vs_line_length', {}), 
        'vulnerability'
    )
    
    return create_common_heatmap(
        data, lengths, lines,
        "Vulnerability Heatmap",
        cmap='YlOrRd',
        value_label='Vulnerability Score',
        ball_counts=ball_counts
    )

def create_bowler_economy_heatmap(line_length_stats):
    """Create economy rate heatmap for bowler analysis"""
    data, ball_counts, lines, lengths = process_line_length_data(
        line_length_stats, 
        'economy'
    )
    
    return create_common_heatmap(
        data, lengths, lines,
        "Economy Rate by Line & Length",
        cmap='YlOrRd',
        value_label='Economy Rate (runs/over)',
        ball_counts=ball_counts
    )

def create_bowler_strike_rate_heatmap(line_length_stats):
    """Create bowling strike rate heatmap for bowler analysis"""
    data, ball_counts, lines, lengths = process_line_length_data(
        line_length_stats, 
        'strike_rate'
    )
    
    return create_common_heatmap(
        data, lengths, lines,
        "Bowling Strike Rate by Line & Length",
        cmap='YlOrRd',
        value_label='Bowling Strike Rate (balls/wicket)',
        ball_counts=ball_counts
    )

def create_phase_vulnerability_heatmap(phase_line_length_stats):
    """Create a heatmap showing a batter's vulnerability in a specific phase"""
    data, ball_counts, lines, lengths = process_line_length_data(
        phase_line_length_stats, 
        'vulnerability'
    )
    
    return create_common_heatmap(
        data, lengths, lines,
        "Phase-wise Vulnerability Heatmap",
        cmap='YlOrRd',
        value_label='Vulnerability Score',
        ball_counts=ball_counts
    )

def create_style_vulnerability_heatmap(style_line_length_stats):
    """Create a heatmap showing a batter's vulnerability against a specific bowling style"""
    data, ball_counts, lines, lengths = process_line_length_data(
        style_line_length_stats, 
        'vulnerability'
    )
    
    return create_common_heatmap(
        data, lengths, lines,
        "Style-wise Vulnerability Heatmap",
        cmap='YlOrRd',
        value_label='Vulnerability Score',
        ball_counts=ball_counts
    )

# Field positions for visualization
FIELD_POSITIONS = {
    'wicketkeeper': (0, -0.25),
    'slip': (0.1, -0.25),
    'gully': (0.2, -0.2),
    'point': (0.4, 0),
    'cover': (0.3, 0.3),
    'mid-off': (0.1, 0.4),
    'mid-on': (-0.1, 0.4),
    'midwicket': (-0.3, 0.3),
    'square leg': (-0.4, 0),
    'fine leg': (-0.2, -0.2),
    'third man': (0.5, -0.5),
    'deep cover': (0.5, 0.5),
    'long-off': (0, 0.8),
    'long-on': (0, 0.8),
    'deep midwicket': (-0.5, 0.5),
    'deep fine leg': (-0.5, -0.5),
    'short leg': (0.05, 0.1),
    'silly point': (0.1, 0.1),
    'leg slip': (-0.1, -0.25)
}

def create_field_placement_visualization(field_setting):
    """Create a visual representation of recommended field placements"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw cricket field
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='green', linewidth=2))
    ax.add_patch(plt.Rectangle((-0.05, -0.2), 0.1, 0.4, fill=True, color='brown'))
    
    # Plot wicketkeeper
    ax.plot(*FIELD_POSITIONS['wicketkeeper'], 'ko', ms=10)
    ax.text(*FIELD_POSITIONS['wicketkeeper'], 'WK', fontsize=8, ha='center', va='bottom')
    
    # Plot field positions
    for position in field_setting.get('catching_positions', []):
        if position in FIELD_POSITIONS:
            ax.plot(*FIELD_POSITIONS[position], 'yo', ms=10)
            ax.text(*FIELD_POSITIONS[position], position, fontsize=8, ha='center', va='bottom')
    
    for position in field_setting.get('boundary_riders', []):
        if position in FIELD_POSITIONS:
            ax.plot(*FIELD_POSITIONS[position], 'bo', ms=10)
            ax.text(*FIELD_POSITIONS[position], position, fontsize=8, ha='center', va='bottom')
    
    # Set plot properties
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title(field_setting.get('description', 'Recommended Field Placement'), fontsize=14)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=l)
        for c, l in [('y', 'Catching Position'), ('b', 'Boundary Rider'), ('k', 'Wicketkeeper')]
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    return fig

def get_strategy_visualization(plan_data):
    """
    Create a summary visualization of a bowling strategy
    
    Parameters:
    -----------
    plan_data : dict
        Dictionary with bowling plan data
    
    Returns:
    --------
    fig : matplotlib figure
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Bowling Strategy for {plan_data['batter']}", fontsize=16)
    
    # 1. Phase-wise recommendations
    phase_names = {1: 'Powerplay', 2: 'Middle', 3: 'Death'}
    phase_data = []
    
    for phase_id, plan in plan_data.get('phase_plans', {}).items():
        if plan.get('line_length_recommendations'):
            rec = plan['line_length_recommendations'][0]
            phase_data.append({
                'phase': phase_names.get(int(phase_id), phase_id),
                'line': rec.get('line', ''),
                'length': rec.get('length', ''),
                'effectiveness': rec.get('effectiveness', 0)
            })
    
    # Create bar chart for phase recommendations
    if phase_data:
        phase_df = pd.DataFrame(phase_data)
        ax = axes[0, 0]
        bars = ax.bar(phase_df['phase'], phase_df['effectiveness'], color='skyblue')
        
        # Add labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{phase_df['line'].iloc[i]} {phase_df['length'].iloc[i]}",
                    ha='center', va='bottom', rotation=0, fontsize=9)
        
        ax.set_title('Phase-specific Recommendations')
        ax.set_ylabel('Effectiveness Score')
    
    # 2. Bowler type effectiveness
    ax = axes[0, 1]
    matchups = plan_data.get('optimal_matchups', [])
    
    if matchups:
        bowler_types = [m['type'] for m in matchups[:3]]
        scores = [m['matchup_score'] for m in matchups[:3]]
        
        # Lower score is better
        ax.barh(bowler_types, scores, color='lightgreen')
        ax.set_title('Optimal Bowler Types')
        ax.set_xlabel('Matchup Score (lower is better)')
        ax.invert_xaxis()  # Invert so best matchup is on top
    
    # 3. Batter weaknesses summary
    ax = axes[1, 0]
    weaknesses = plan_data.get('weaknesses', [])
    
    if weaknesses:
        weakness_text = ""
        for w in weaknesses:
            if w['type'] == 'bowler_type':
                weakness_text += f"• Vulnerable against {w['bowler_type']}\n"
            elif w['type'] == 'line_length':
                weakness_text += f"• Struggles against {w['line']} {w['length']}\n"
            elif w['type'] == 'phase':
                phase_name = phase_names.get(w['phase'], w['phase'])
                weakness_text += f"• Less effective in {phase_name}\n"
        
        ax.text(0.5, 0.5, weakness_text, ha='center', va='center', fontsize=12)
        ax.set_title('Identified Weaknesses')
        ax.axis('off')
    
    # 4. Summary text
    ax = axes[1, 1]
    summary = plan_data.get('summary', '').split('\n')
    
    # Filter to get just the essentials
    essential_lines = [line for line in summary if line.startswith('-') or line.startswith('•')]
    summary_text = '\n'.join(essential_lines)
    
    ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10)
    ax.set_title('Strategy Summary')
    ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
def create_smart_runs_comparison(batsman_data):
    """Create comparison chart for Smart vs Conventional Runs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    matches = len(batsman_data['match_performances'])
    x = np.arange(matches)
    
    # Extract performance data
    conventional = [perf['conventional_runs'] for perf in batsman_data['match_performances']]
    smart = [perf['smart_runs'] for perf in batsman_data['match_performances']]
    
    # Plot bars
    width = 0.35
    ax.bar(x - width/2, conventional, width, label='Conventional Runs', color='skyblue')
    ax.bar(x + width/2, smart, width, label='Smart Runs', color='orange')
    
    # Add labels
    ax.set_xlabel('Match')
    ax.set_ylabel('Runs')
    ax.set_title('Smart vs Conventional Runs Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f"M{i+1}" for i in range(matches)])
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_smart_wickets_comparison(bowler_data):
    """Create comparison chart for Smart vs Conventional Wickets"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    matches = len(bowler_data['match_performances'])
    x = np.arange(matches)
    
    # Extract performance data
    conventional = [perf['conventional_wickets'] for perf in bowler_data['match_performances']]
    smart = [perf['smart_wickets'] for perf in bowler_data['match_performances']]
    
    # Plot bars
    width = 0.35
    ax.bar(x - width/2, conventional, width, label='Conventional Wickets', color='skyblue')
    ax.bar(x + width/2, smart, width, label='Smart Wickets', color='orange')
    
    # Add labels
    ax.set_xlabel('Match')
    ax.set_ylabel('Wickets')
    ax.set_title('Smart vs Conventional Wickets Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f"M{i+1}" for i in range(matches)])
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_key_moments_timeline(key_moments):
    """Create timeline visualization of key match moments"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by over
    moments_sorted = sorted(key_moments, key=lambda x: x['over'])
    
    # Extract data
    overs = [moment['over'] for moment in moments_sorted]
    impacts = [moment['total_impact'] for moment in moments_sorted]
    players = [moment['player'] for moment in moments_sorted]
    
    # Plot scatter with color based on impact
    colors = ['green' if impact > 0 else 'red' for impact in impacts]
    ax.scatter(overs, impacts, s=100, alpha=0.7, c=colors)
    
    # Add player labels
    for i, player in enumerate(players):
        ax.annotate(player, 
                   (overs[i], impacts[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add match progression line
    ax.plot(overs, impacts, 'k--', alpha=0.3)
    
    # Set labels and grid
    ax.set_xlabel('Over')
    ax.set_ylabel('Impact')
    ax.set_title('Key Match Moments')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Positive Impact'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Negative Impact')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

def create_player_impact_breakdown(player_impacts):
    """Create stacked bar chart showing batting vs bowling impact"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    players = [impact['player'] for impact in player_impacts]
    batting_impacts = [impact['batting_impact'] for impact in player_impacts]
    bowling_impacts = [impact['bowling_impact'] for impact in player_impacts]
    
    # Plot stacked bars
    ax.bar(players, batting_impacts, label='Batting Impact', color='skyblue')
    ax.bar(players, bowling_impacts, bottom=batting_impacts, label='Bowling Impact', color='lightgreen')
    
    # Set labels and legend
    ax.set_xlabel('Player')
    ax.set_ylabel('Impact Score')
    ax.set_title('Player Impact Breakdown')
    ax.set_xticks(range(len(players)))
    ax.set_xticklabels(players, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_impact_radar_chart(player_data):
    """Create radar chart for player impact metrics"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Categories for radar chart
    categories = ['Batting Impact', 'Bowling Impact', 'Match Influence', 
                 'Pressure Handling', 'Consistency']
    N = len(categories)
    
    # Values (normalized to 0-1 range)
    values = [
        player_data.get('batting_impact', 0) / 5,
        player_data.get('bowling_impact', 0) / 5,
        player_data.get('total_impact', 0) / 10,
        0.5,  # Placeholder - would calculate from pressure index data
        0.7   # Placeholder - would calculate from performance consistency
    ]
    
    # Close the loop
    values = values + [values[0]]
    
    # Compute angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    
    # Fill area
    ax.fill(angles, values, 'skyblue', alpha=0.4)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Remove radial labels
    ax.set_yticklabels([])
    
    # Add title
    plt.title(f"Impact Profile: {player_data.get('player', 'Player')}")
    
    return fig