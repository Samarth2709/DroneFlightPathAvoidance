"""
Global configuration parameters for the Drone Flight Path Avoidance simulation.
This file contains all the constants used throughout the program.
"""
import sys

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

# Default filename constants
DEFAULT_MAP_FILE = "3d_map.pkl"
DEFAULT_IMG_FILE = "3d_map.png"

# Error messages
ERROR_FILE_NOT_FOUND = "Map file '{}' not found"
ERROR_MAP_DATA_MISSING_KEY = "Map data is missing required key: {}"
ERROR_INVALID_DIMENSIONS = "Invalid dimensions: {}"
ERROR_NO_CYLINDERS = "No cylinders were generated. Try increasing the map size or decreasing the minimum distance constraints."
ERROR_MAP_GENERATION_TIMEOUT = "Map generation timed out. Try adjusting the parameters."

# Validation function to ensure configuration values are valid
def validate_config():
    """Validates all configuration parameters and raises ValueError for invalid values"""
    errors = []
    
    # Check map dimensions
    if MAP_SIZE <= 0:
        errors.append(f"MAP_SIZE must be positive (got {MAP_SIZE})")
    
    # Check cylinder parameters
    if NUM_CYLINDERS < 0:
        errors.append(f"NUM_CYLINDERS cannot be negative (got {NUM_CYLINDERS})")
    if MIN_CYLINDER_HEIGHT < 0:
        errors.append(f"MIN_CYLINDER_HEIGHT cannot be negative (got {MIN_CYLINDER_HEIGHT})")
    if MAX_CYLINDER_HEIGHT <= MIN_CYLINDER_HEIGHT:
        errors.append(f"MAX_CYLINDER_HEIGHT must be greater than MIN_CYLINDER_HEIGHT ({MAX_CYLINDER_HEIGHT} <= {MIN_CYLINDER_HEIGHT})")
    if MIN_CYLINDER_RADIUS < 0:
        errors.append(f"MIN_CYLINDER_RADIUS cannot be negative (got {MIN_CYLINDER_RADIUS})")
    if MAX_CYLINDER_RADIUS <= MIN_CYLINDER_RADIUS:
        errors.append(f"MAX_CYLINDER_RADIUS must be greater than MIN_CYLINDER_RADIUS ({MAX_CYLINDER_RADIUS} <= {MIN_CYLINDER_RADIUS})")
        
    # Check placement constraints
    if MIN_DISTANCE_FROM_POINTS < 0:
        errors.append(f"MIN_DISTANCE_FROM_POINTS cannot be negative (got {MIN_DISTANCE_FROM_POINTS})")
    if MIN_DESTINATION_DISTANCE < 0:
        errors.append(f"MIN_DESTINATION_DISTANCE cannot be negative (got {MIN_DESTINATION_DISTANCE})")
    if MAX_DESTINATION_DISTANCE <= MIN_DESTINATION_DISTANCE:
        errors.append(f"MAX_DESTINATION_DISTANCE must be greater than MIN_DESTINATION_DISTANCE ({MAX_DESTINATION_DISTANCE} <= {MIN_DESTINATION_DISTANCE})")
    if MAX_DESTINATION_DISTANCE > MAP_SIZE * 1.414:  # sqrt(2) * MAP_SIZE is the max possible distance
        errors.append(f"MAX_DESTINATION_DISTANCE ({MAX_DESTINATION_DISTANCE}) exceeds the maximum possible distance in the map ({MAP_SIZE * 1.414:.2f})")
        
    # Check animation parameters
    if ANIMATION_DURATION <= 0:
        errors.append(f"ANIMATION_DURATION must be positive (got {ANIMATION_DURATION})")
    if FPS <= 0:
        errors.append(f"FPS must be positive (got {FPS})")
    
    # Check opacity values
    if not 0 <= CYLINDER_ALPHA <= 1:
        errors.append(f"CYLINDER_ALPHA must be between 0 and 1 (got {CYLINDER_ALPHA})")
    if not 0 <= CYLINDER_LINE_ALPHA <= 1:
        errors.append(f"CYLINDER_LINE_ALPHA must be between 0 and 1 (got {CYLINDER_LINE_ALPHA})")
    
    # Check space constraint
    if MIN_DISTANCE_FROM_POINTS * 2 + MAX_CYLINDER_RADIUS * 2 > MAP_SIZE:
        errors.append(f"Map may be too constrained: MIN_DISTANCE_FROM_POINTS*2 + MAX_CYLINDER_RADIUS*2 ({MIN_DISTANCE_FROM_POINTS*2 + MAX_CYLINDER_RADIUS*2}) > MAP_SIZE ({MAP_SIZE})")
    
    # Report all errors if any
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    return True

# Validate the configuration on import
try:
    validate_config()
except ValueError as e:
    print(f"Error in configuration: {e}", file=sys.stderr)
    sys.exit(1)
