import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import io

def create_vulnerability_heatmap(batter_data):
    """
    Create a heatmap showing a batter's vulnerability to different line/length combinations
    
    Parameters:
    -----------
    batter_data : dict
        Dictionary with batter analysis data
    
    Returns:
    --------
    fig : matplotlib figure
    """
    # Create a 3x4 grid for line and length combinations
    # Lines: Off, Middle, Leg
    # Lengths: Yorker, Full, Good, Short
    
    # Extract line-length data
    line_length_stats = batter_data.get('vs_line_length', {})
    
    # Create a data frame for the heatmap
    lines = ['Off', 'Middle', 'Leg']
    lengths = ['Yorker', 'Full', 'Good', 'Short']
    
    data = np.zeros((len(lengths), len(lines)))
    count = np.zeros((len(lengths), len(lines)))
    
    for (line, length), stats in line_length_stats.items():
        if line in lines and length in lengths:
            row_idx = lengths.index(length)
            col_idx = lines.index(line)
            
            # Calculate vulnerability score
            # Lower average = higher vulnerability
            if stats['dismissals'] > 0 and stats['balls'] >= 5:
                vulnerability = 100 / stats['average']
            else:
                # If no dismissals, use strike rate (lower = more vulnerable)
                vulnerability = 100 / stats['strike_rate'] if stats['strike_rate'] > 0 else 0
            
            data[row_idx, col_idx] = vulnerability
            count[row_idx, col_idx] = stats['balls']
    
    # Normalize data for better visualization
    max_vulnerability = np.max(data) if np.max(data) > 0 else 1
    data = data / max_vulnerability
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=lines,
        yticklabels=lengths,
        ax=ax
    )
    
    # Add title and labels
    ax.set_title(f"Vulnerability Heatmap", fontsize=14)
    ax.set_xlabel("Line", fontsize=12)
    ax.set_ylabel("Length", fontsize=12)
    
    # Return the figure
    return fig

def create_field_placement_visualization(field_setting):
    """
    Create a visual representation of recommended field placements
    
    Parameters:
    -----------
    field_setting : dict
        Dictionary with field placement recommendations
    
    Returns:
    --------
    fig : matplotlib figure
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw cricket field (circle)
    circle = plt.Circle((0, 0), 1, fill=False, color='green', linewidth=2)
    ax.add_patch(circle)
    
    # Draw pitch rectangle
    rect = plt.Rectangle((-0.05, -0.2), 0.1, 0.4, fill=True, color='brown')
    ax.add_patch(rect)
    
    # Field positions
    field_positions = {
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
    
    # Always show wicketkeeper
    ax.plot(*field_positions['wicketkeeper'], 'ko', ms=10)
    ax.text(*field_positions['wicketkeeper'], 'WK', fontsize=8, ha='center', va='bottom')
    
    # Show catching positions
    for position in field_setting.get('catching_positions', []):
        if position in field_positions:
            ax.plot(*field_positions[position], 'yo', ms=10)
            ax.text(*field_positions[position], position, fontsize=8, ha='center', va='bottom')
    
    # Show boundary riders
    for position in field_setting.get('boundary_riders', []):
        if position in field_positions:
            ax.plot(*field_positions[position], 'bo', ms=10)
            ax.text(*field_positions[position], position, fontsize=8, ha='center', va='bottom')
    
    # Set plot properties
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    plt.title(field_setting.get('description', 'Recommended Field Placement'), fontsize=14)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='y', markersize=10, label='Catching Position'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Boundary Rider'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=10, label='Wicketkeeper')
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