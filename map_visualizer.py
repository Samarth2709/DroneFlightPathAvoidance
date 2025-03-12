import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import argparse
import os

def load_map(filename):
    """
    Load a saved 3D map from a pickle file.
    
    Args:
        filename: Path to the pickle file
        
    Returns:
        Dictionary containing map data
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Map file '{filename}' not found")
    
    with open(filename, 'rb') as f:
        map_data = pickle.load(f)
    
    # Validate that the required keys are present
    required_keys = ["map_size", "origin", "destination", "cylinders"]
    for key in required_keys:
        if key not in map_data:
            raise ValueError(f"Map data is missing required key: {key}")
    
    return map_data

def visualize_map(map_data, output_file=None, show=True):
    """
    Visualize a 3D map from the provided map data.
    
    Args:
        map_data: Dictionary containing map information
        output_file: If provided, save the visualization to this file
        show: Whether to display the visualization
        
    Returns:
        The matplotlib figure object
    """
    map_size = map_data["map_size"]
    origin = map_data["origin"]
    destination = map_data["destination"]
    cylinders = map_data["cylinders"]
    
    # Determine max height for plot limits
    max_height = max([h for _, _, _, h in cylinders]) if cylinders else 50
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set the limits of the plot
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_zlim(0, max_height + 10)
    
    # Create a base map (ground)
    x = np.linspace(0, map_size, 100)
    y = np.linspace(0, map_size, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Plot the ground as a surface
    ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.3, zorder=1)
    
    # Plot cylinders
    for x, y, radius, height in cylinders:
        # Create a circle in the x-y plane
        theta = np.linspace(0, 2 * np.pi, 32)
        circle_x = radius * np.cos(theta) + x
        circle_y = radius * np.sin(theta) + y
        
        # Create cylinder
        # Bottom circle
        ax.plot(circle_x, circle_y, np.zeros_like(theta), color='grey', alpha=0.4)
        # Top circle
        ax.plot(circle_x, circle_y, np.ones_like(theta) * height, color='grey', alpha=0.4)
        
        # Connect bottom and top circles with lines
        for i in range(0, len(theta), 4):
            ax.plot([circle_x[i], circle_x[i]], [circle_y[i], circle_y[i]], 
                    [0, height], color='grey', alpha=0.4)
        
        # Create cylinder surface
        z = np.linspace(0, height, 10)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + x
        y_grid = radius * np.sin(theta_grid) + y
        
        ax.plot_surface(x_grid, y_grid, z_grid, color='grey', alpha=0.3)
    
    # Plot origin point (red) and destination point (green)
    ax.scatter(origin[0], origin[1], origin[2], color='red', s=100, marker='o', label='Origin')
    ax.scatter(destination[0], destination[1], destination[2], color='green', s=100, marker='o', label='Destination')
    
    # Add a direct line from origin to destination
    ax.plot([origin[0], destination[0]], 
            [origin[1], destination[1]], 
            [origin[2], destination[2]], 'k--', alpha=0.5)
    
    # Calculate and display the distance
    distance = np.sqrt((origin[0]-destination[0])**2 + (origin[1]-destination[1])**2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Map with {len(cylinders)} Cylinders\nDistance: {distance:.2f} units')
    ax.legend()
    
    # Adjust view angle
    ax.view_init(elev=30, azim=45)
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    # Show the plot if requested
    if show:
        plt.show()
    
    return fig

def display_map_info(map_data):
    """Print information about the loaded map"""
    print("\nMap Information:")
    print(f"Map size: {map_data['map_size']} x {map_data['map_size']}")
    print(f"Origin: ({map_data['origin'][0]:.2f}, {map_data['origin'][1]:.2f}, {map_data['origin'][2]:.2f})")
    print(f"Destination: ({map_data['destination'][0]:.2f}, {map_data['destination'][1]:.2f}, {map_data['destination'][2]:.2f})")
    
    # Calculate distance
    origin = map_data['origin']
    destination = map_data['destination']
    distance = np.sqrt((origin[0]-destination[0])**2 + (origin[1]-destination[1])**2)
    print(f"Distance between origin and destination: {distance:.2f} units")
    
    print(f"Number of cylinders: {len(map_data['cylinders'])}")
    
    # Calculate cylinder statistics
    if map_data['cylinders']:
        heights = [h for _, _, _, h in map_data['cylinders']]
        radii = [r for _, _, r, _ in map_data['cylinders']]
        
        print(f"Cylinder height range: {min(heights):.2f} to {max(heights):.2f} units")
        print(f"Cylinder radius range: {min(radii):.2f} to {max(radii):.2f} units")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Visualize a 3D map from a pickle file')
    parser.add_argument('filename', help='Path to the pickle file containing the map data')
    parser.add_argument('--output', '-o', help='Save the visualization to this file')
    parser.add_argument('--no-display', action='store_true', help='Do not display the visualization')
    parser.add_argument('--info-only', action='store_true', help='Only display map information, do not visualize')
    
    args = parser.parse_args()
    
    try:
        # Load the map data
        map_data = load_map(args.filename)
        
        # Display map information
        display_map_info(map_data)
        
        # Visualize the map if not in info-only mode
        if not args.info_only:
            visualize_map(map_data, output_file=args.output, show=not args.no_display)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()