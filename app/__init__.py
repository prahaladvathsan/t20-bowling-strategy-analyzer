"""T20 Bowling Strategy Analyzer Application Package"""
__version__ = "1.0.0"
__author__ = "Your Name"

# Make the main function available at the package level
# but import it only when needed to avoid circular imports
def main():
    from app.app import main as _main
    return _main()