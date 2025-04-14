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
    
    # Create heatmap using imshow
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
            
            # For infinite values, always use black text
            text_color = "black" if (np.isinf(value) or value < np.nanmax(display_data) * 0.7) else "white"
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
    
    # Replace inf values with nan for normalization
    data_for_norm = np.copy(data)
    data_for_norm[np.isinf(data_for_norm)] = np.nan
    
    # Normalize vulnerability scores
    max_value = np.nanmax(data_for_norm) if np.nanmax(data_for_norm) > 0 else 1
    normalized_data = data.copy()  # Keep inf values in the actual data
    mask = ~np.isinf(normalized_data)  # Only normalize non-inf values
    normalized_data[mask] = normalized_data[mask] / max_value if max_value > 0 else normalized_data[mask]
    
    return create_common_heatmap(
        normalized_data, lengths, lines,
        "Vulnerability Heatmap",
        cmap='YlOrRd',
        value_label='Normalized Vulnerability Score',
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