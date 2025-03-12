"""
Global configuration parameters for the Drone Flight Path Avoidance simulation.
This file contains all the constants used throughout the program.
"""

# Map dimensions
MAP_SIZE = 100  # Size of the square map (x and y dimensions)

# Obstacle (cylinder) parameters
NUM_CYLINDERS = 30       # Number of cylinders to generate
MIN_CYLINDER_HEIGHT = 2  # Minimum height of cylinders
MAX_CYLINDER_HEIGHT = 30  # Maximum height of cylinders
MIN_CYLINDER_RADIUS = 2  # Minimum radius of cylinders
MAX_CYLINDER_RADIUS = 5  # Maximum radius of cylinders

# Placement constraints
MIN_DISTANCE_FROM_POINTS = 10  # Minimum distance cylinders must be from origin/destination
MIN_DESTINATION_DISTANCE = 70  # Minimum distance between origin and destination
MAX_DESTINATION_DISTANCE = 120  # Maximum distance between origin and destination

# Visualization parameters
GROUND_COLOR = 'lightgray'  # Color of the ground surface
CYLINDER_COLOR = 'grey'      # Color of the cylinders
CYLINDER_ALPHA = 0.3         # Opacity of cylinder surfaces
CYLINDER_LINE_ALPHA = 0.4    # Opacity of cylinder outlines
ORIGIN_COLOR = 'red'         # Color of the origin point
DESTINATION_COLOR = 'green'  # Color of the destination point
PLANNED_PATH_COLOR = 'black' # Color of the planned path line (dashed)
ACTUAL_PATH_COLOR = 'blue'   # Color of the actual path line (solid)

# Animation parameters
ANIMATION_DURATION = 5.0     # Duration of flight path animation in seconds
FPS = 20                     # Frames per second for animation