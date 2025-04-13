import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import io

def create_common_heatmap(data, row_labels, col_labels, title, cmap='YlOrRd', value_label='Value'):
    """
    Common function to create heatmaps with consistent styling
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D array of values to display
    row_labels : list
        Labels for rows (y-axis)
    col_labels : list
        Labels for columns (x-axis)
    title : str
        Title of the heatmap
    cmap : str
        Matplotlib colormap name
    value_label : str
        Label for the colorbar
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Create figure and axes with specified size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap using imshow
    im = ax.imshow(data, cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(value_label, rotation=-90, va="bottom")

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = data[i, j]
            text = ax.text(j, i, f'{value:.2f}',
                         ha="center", va="center",
                         color="black" if value < np.max(data) * 0.7 else "white",
                         fontweight='bold')

    # Configure axis labels and ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate and align the tick labels so they look better
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add title and labels
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Line", fontsize=12)
    ax.set_ylabel("Length", fontsize=12)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig

def create_vulnerability_heatmap(batter_data):
    """
    Create a heatmap showing a batter's vulnerability to different line/length combinations
    """
    from src.data_processor import DataProcessor
    
    # Define display labels in the same order as numeric indices
    lines = [DataProcessor.LINE_DISPLAY[i] for i in range(5)]
    lengths = [DataProcessor.LENGTH_DISPLAY[i] for i in range(6)]
    
    # Initialize data matrix
    data = np.zeros((6, 5))  # 6 lengths x 5 lines
    
    # Extract line-length data
    line_length_stats = batter_data.get('vs_line_length', {})
    
    # Fill in the data where we have values
    if line_length_stats:
        for (line, length), stats in line_length_stats.items():
            try:
                # Find indices based on display values
                col_idx = lines.index(line)
                row_idx = lengths.index(length)
                
                # Calculate vulnerability score
                if stats['dismissals'] > 0:
                    vulnerability = 100 / stats['average']
                else:
                    # If no dismissals, use strike rate (lower = more vulnerable)
                    vulnerability = 100 / stats['strike_rate'] if stats['strike_rate'] > 0 else 0
                
                data[row_idx, col_idx] = vulnerability
            except (ValueError, IndexError):
                continue
    
    # Normalize data for better visualization
    max_vulnerability = np.max(data) if np.max(data) > 0 else 1
    normalized_data = data / max_vulnerability if max_vulnerability > 0 else data
    
    # Use common heatmap function
    return create_common_heatmap(
        normalized_data,
        lengths,
        lines,
        "Vulnerability Heatmap",
        cmap='YlOrRd',
        value_label='Normalized Vulnerability Score'
    )

def create_bowler_economy_heatmap(line_length_stats):
    """Create economy rate heatmap for bowler analysis"""
    from src.data_processor import DataProcessor
    
    # Define display labels in the same order as numeric indices
    lines = [DataProcessor.LINE_DISPLAY[i] for i in range(5)]
    lengths = [DataProcessor.LENGTH_DISPLAY[i] for i in range(6)]
    
    # Initialize data matrix
    data = np.zeros((len(lengths), len(lines)))
    
    # Fill in the data
    for (line, length), stats in line_length_stats.items():
        try:
            # Find indices based on display values
            col_idx = lines.index(line)
            row_idx = lengths.index(length)
            data[row_idx, col_idx] = stats['economy']
        except (ValueError, IndexError):
            continue
    
    return create_common_heatmap(
        data,
        lengths,
        lines,
        "Economy Rate by Line & Length",
        cmap='YlOrRd',
        value_label='Economy Rate'
    )

def create_bowler_strike_rate_heatmap(line_length_stats):
    """Create strike rate heatmap for bowler analysis"""
    from src.data_processor import DataProcessor
    
    # Define display labels in the same order as numeric indices
    lines = [DataProcessor.LINE_DISPLAY[i] for i in range(5)]
    lengths = [DataProcessor.LENGTH_DISPLAY[i] for i in range(6)]
    
    # Initialize data matrix
    data = np.zeros((len(lengths), len(lines)))
    
    # Fill in the data
    for (line, length), stats in line_length_stats.items():
        try:
            if stats['wickets'] > 0:
                # Find indices based on display values
                col_idx = lines.index(line)
                row_idx = lengths.index(length)
                data[row_idx, col_idx] = stats['strike_rate']
        except (ValueError, IndexError):
            continue
    
    return create_common_heatmap(
        data,
        lengths,
        lines,
        "Strike Rate by Line & Length",
        cmap='YlOrRd',
        value_label='Strike Rate'
    )

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