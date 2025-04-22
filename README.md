# T20 Bowling Strategy Analyzer

A data-driven application to analyze T20 cricket data and generate optimal bowling strategies.

## Features

- **Batter Analysis**: Analyze batter weaknesses and strengths against different bowling types and phases
- **Bowler Strategies**: Review and optimize bowler performance and strategies
- **Match-up Optimizer**: Find optimal bowler-batter matchups
- **Complete Bowling Plan**: Generate comprehensive bowling plans for specific batters

## Project Structure

```
app/
├── app.py                # Main entry point (lightweight)
├── components/          # Reusable UI components
│   ├── metrics.py       # Common metric displays
│   ├── visualizations.py # Wrapper for visualization functions
│   └── selectors.py     # Common selection widgets
├── pages/              # Individual page implementations
│   ├── batter_analysis.py
│   ├── bowler_strategies.py
│   ├── matchup_optimizer.py
│   └── bowling_plan.py
├── utils/              # Utility functions
│   ├── visualization.py
│   └── state_management.py
└── config.py           # Configuration constants
```

## Setup Instructions

1. Install Python 3.8+ and pip
2. Clone the repository
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app/app.py
   ```

## Key Components

- **Lightweight Main App**: The main app.py only handles initialization and routing
- **Modular Pages**: Each analysis type has its own dedicated page module
- **Reusable Components**: Common UI elements are extracted into reusable components
- **State Management**: Centralized approach using Streamlit's session state
- **Configuration**: Constants and settings are centralized in config.py

## Data Sources

- `data/t20_bbb.csv`: Ball-by-ball T20 match data
- `db/`: Processed data and analysis results
  - batter_profiles.json
  - bowler_profiles.json
  - plan_generator_data.json
  - processed_data.parquet

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See LICENSE file for details
