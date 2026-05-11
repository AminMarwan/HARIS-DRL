import os

from plotting import generate_all_figures


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, "outputs", "logs")
FIGURE_DIR = os.path.join(ROOT_DIR, "outputs", "figures")


if __name__ == "__main__":
    generate_all_figures(LOG_DIR, FIGURE_DIR)
    print(f"Figures saved to: {FIGURE_DIR}")
