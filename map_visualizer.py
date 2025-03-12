import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import argparse
import os
import sys
import time
from matplotlib import cm

# Import global configuration parameters
from config import *

def load_map(filename):
    """
    Load a saved 3D map from a pickle file.
    
    Args:
        filename: Path to the pickle file
        
    Returns:
        Dictionary containing map data
        
    Raises:
        FileNotFoundError: If the map file doesn't exist
        ValueError: If the map data is invalid or missing required keys
        Exception: For any other loading errors
    """
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(ERROR_FILE_NOT_FOUND.format(filename))
    
    # Validate file has proper extension
    if not filename.lower().endswith('.pkl'):
        print(f"Warning: File '{filename}' doesn't have the expected .pkl extension")
    
    try:
        # Attempt to load the map data
        with open(filename, 'rb') as f:
            map_data = pickle.load(f)
        
        # Validate that the required keys are present
        required_keys = ["map_size", "origin", "destination", "cylinders"]
        for key in required_keys:
            if key not in map_data:
                raise ValueError(ERROR_MAP_DATA_MISSING_KEY.format(key))
        
        # Validate data types and values
        if not isinstance(map_data["map_size"], (int, float)) or map_data["map_size"] <= 0:
            raise ValueError(ERROR_INVALID_DIMENSIONS.format(f"map_size must be positive, got {map_data['map_size']}"))
        
        if not isinstance(map_data["origin"], np.ndarray) and not isinstance(map_data["origin"], list):
            raise ValueError("Origin must be a numpy array or list")
        
        if not isinstance(map_data["destination"], np.ndarray) and not isinstance(map_data["destination"], list):
            raise ValueError("Destination must be a numpy array or list")
        
        if len(map_data["origin"]) != 3 or len(map_data["destination"]) != 3:
            raise ValueError("Origin and destination must be 3D points")
        
        # Ensure cylinders exist
        if not map_data["cylinders"]:
            print("Warning: Map contains no cylinders")
        
        # Convert lists to numpy arrays if necessary
        if isinstance(map_data["origin"], list):
            map_data["origin"] = np.array(map_data["origin"])
        
        if isinstance(map_data["destination"], list):
            map_data["destination"] = np.array(map_data["destination"])
        
        return map_data
        
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Error unpickling map file: {e}. The file might be corrupted or not a valid pickle file.")
    except Exception as e:
        raise Exception(f"Error loading map file: {e}")

def visualize_map(map_data, output_file=None, show=True, heatmap=False):
    """
    Visualize a 3D map from the provided map data.
    
    Args:
        map_data: Dictionary containing map information
        output_file: If provided, save the visualization to this file
        show: Whether to display the visualization
        heatmap: If True, show a heatmap of cylinder heights
        
    Returns:
        The matplotlib figure object
        
    Raises:
        ValueError: If map data is invalid
        IOError: If output file cannot be saved
    """
    try:
        # Extract data from the map
        map_size = map_data["map_size"]
        origin = map_data["origin"]
        destination = map_data["destination"]
        cylinders = map_data["cylinders"]
        
        # Determine max height for plot limits
        max_height = max([h for _, _, _, h in cylinders]) if cylinders else 50
        buffer = max(10, max_height * 0.2)  # Add buffer space
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set the limits of the plot
        ax.set_xlim(0, map_size)
        ax.set_ylim(0, map_size)
        ax.set_zlim(0, max_height + buffer)
        
        # Create a base map (ground)
        x = np.linspace(0, map_size, 100)
        y = np.linspace(0, map_size, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Plot the ground as a surface
        ax.plot_surface(X, Y, Z, color=GROUND_COLOR, alpha=0.3, zorder=1)
        
        # Plot cylinders
        for x, y, radius, height in cylinders:
            # Create a circle in the x-y plane
            theta = np.linspace(0, 2 * np.pi, 32)
            circle_x = radius * np.cos(theta) + x
            circle_y = radius * np.sin(theta) + y
            
            # Calculate color based on height if heatmap is enabled
            if heatmap:
                # Normalize height to [0, 1]
                norm_height = (height - MIN_CYLINDER_HEIGHT) / (max_height - MIN_CYLINDER_HEIGHT) if max_height > MIN_CYLINDER_HEIGHT else 0.5
                # Get color from colormap
                color = cm.viridis(norm_height)
                alpha = 0.6  # Higher alpha for heatmap mode
            else:
                color = CYLINDER_COLOR
                alpha = CYLINDER_ALPHA
            
            # Create cylinder
            # Bottom circle
            ax.plot(circle_x, circle_y, np.zeros_like(theta), color=color, alpha=CYLINDER_LINE_ALPHA)
            # Top circle
            ax.plot(circle_x, circle_y, np.ones_like(theta) * height, color=color, alpha=CYLINDER_LINE_ALPHA)
            
            # Connect bottom and top circles with lines
            for i in range(0, len(theta), 4):
                ax.plot([circle_x[i], circle_x[i]], [circle_y[i], circle_y[i]], 
                        [0, height], color=color, alpha=CYLINDER_LINE_ALPHA)
            
            # Create cylinder surface
            z = np.linspace(0, height, 10)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = radius * np.cos(theta_grid) + x
            y_grid = radius * np.sin(theta_grid) + y
            
            ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)
        
        # Plot origin point (red) and destination point (green)
        ax.scatter(origin[0], origin[1], origin[2], color=ORIGIN_COLOR, s=100, marker='o', label='Origin')
        ax.scatter(destination[0], destination[1], destination[2], color=DESTINATION_COLOR, s=100, marker='o', label='Destination')
        
        # Add a direct line from origin to destination
        ax.plot([origin[0], destination[0]], 
                [origin[1], destination[1]], 
                [origin[2], destination[2]], '--', color=PLANNED_PATH_COLOR, alpha=0.5)
        
        # Calculate and display the distance
        distance = np.sqrt((origin[0]-destination[0])**2 + (origin[1]-destination[1])**2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set title with info
        if heatmap:
            ax.set_title(f'3D Map with {len(cylinders)} Cylinders (Height Heatmap)\nDistance: {distance:.2f} units')
        else:
            ax.set_title(f'3D Map with {len(cylinders)} Cylinders\nDistance: {distance:.2f} units')
        
        ax.legend()
        
        # Adjust view angle
        ax.view_init(elev=30, azim=45)
        
        # Save figure if output file is specified
        if output_file:
            try:
                # Ensure the directory exists
                directory = os.path.dirname(os.path.abspath(output_file))
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Visualization saved to {output_file}")
            except (IOError, OSError) as e:
                print(f"Warning: Could not save visualization to {output_file}: {str(e)}")
        
        # Show the plot if requested
        if show:
            plt.show()
        
        # Close the figure to free resources
        plt.close(fig)
        return fig
        
    except Exception as e:
        # Make sure to close the figure if an error occurs
        plt.close()
        raise e

def display_map_info(map_data):
    """
    Print information about the loaded map
    
    Args:
        map_data: Dictionary containing map information
    """
    try:
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
            
            # Additional statistics
            avg_height = sum(heights) / len(heights)
            avg_radius = sum(radii) / len(radii)
            print(f"Average cylinder height: {avg_height:.2f} units")
            print(f"Average cylinder radius: {avg_radius:.2f} units")
            
            # Distribution of cylinders
            height_quartiles = np.percentile(heights, [25, 50, 75])
            print(f"Height quartiles: {height_quartiles[0]:.2f}, {height_quartiles[1]:.2f}, {height_quartiles[2]:.2f}")
            
    except KeyError as e:
        print(f"Error displaying map information: Missing key {e}")
    except Exception as e:
        print(f"Error displaying map information: {e}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Visualize a 3D map from a pickle file')
    parser.add_argument('filename', help='Path to the pickle file containing the map data')
    parser.add_argument('--output', '-o', help='Save the visualization to this file')
    parser.add_argument('--no-display', action='store_true', help='Do not display the visualization')
    parser.add_argument('--info-only', action='store_true', help='Only display map information, do not visualize')
    parser.add_argument('--heatmap', action='store_true', help='Show cylinders as a heatmap by height')
    parser.add_argument('--view-angle', type=str, default='30,45', help='View angle (elevation,azimuth) for 3D plot')
    
    args = parser.parse_args()
    
    try:
        # Load the map data
        start_time = time.time()
        map_data = load_map(args.filename)
        loading_time = time.time() - start_time
        
        print(f"Map loaded in {loading_time:.2f} seconds")
        
        # Display map information
        display_map_info(map_data)
        
        # Visualize the map if not in info-only mode
        if not args.info_only:
            # Parse view angle if provided
            view_elev, view_azim = 30, 45  # Default view angles
            if args.view_angle:
                try:
                    angle_parts = args.view_angle.split(',')
                    if len(angle_parts) == 2:
                        view_elev = float(angle_parts[0])
                        view_azim = float(angle_parts[1])
                except ValueError:
                    print(f"Warning: Invalid view angle format: {args.view_angle}. Using defaults.")
            
            # Start visualization timer
            start_time = time.time()
            
            # Create visualization
            fig = visualize_map(map_data, output_file=args.output, show=not args.no_display, heatmap=args.heatmap)
            
            # Calculate visualization time
            vis_time = time.time() - start_time
            if not args.no_display:
                print(f"Visualization created in {vis_time:.2f} seconds")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please check that the file '{args.filename}' exists.")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()