import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import sys
import time
import argparse

# Import global configuration parameters
from config import *

def load_map_data(pkl_file=DEFAULT_MAP_FILE):
    """
    Load map data (map_size, origin, destination, cylinders) from a pickle file.
    
    Args:
        pkl_file: Path to the pickle file containing map data
        
    Returns:
        dict: Map data with keys: "map_size", "origin", "destination", "cylinders"
        
    Raises:
        FileNotFoundError: If the map file doesn't exist
        ValueError: If the map data is invalid or missing required keys
        Exception: For any other loading errors
    """
    # Check if file exists
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(ERROR_FILE_NOT_FOUND.format(pkl_file))
    
    try:
        # Attempt to load the map data
        with open(pkl_file, 'rb') as f:
            map_data = pickle.load(f)
        
        # Validate that the required keys are present
        required_keys = ["map_size", "origin", "destination", "cylinders"]
        for key in required_keys:
            if key not in map_data:
                raise ValueError(ERROR_MAP_DATA_MISSING_KEY.format(key))
        
        # Validate the map has cylinders
        if not map_data["cylinders"]:
            print("Warning: Map contains no cylinders, animation may look incomplete")
        
        return map_data
        
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Error unpickling map file: {e}. The file might be corrupted or not a valid pickle file.")
    except Exception as e:
        raise Exception(f"Error loading map file: {e}")

def plot_cylinders(ax, cylinders):
    """
    Plot cylinders in the 3D axis.
    
    Args:
        ax: Matplotlib 3D axis object
        cylinders: List of cylinders, each defined as (x_center, y_center, radius, height)
        
    Raises:
        ValueError: If cylinder data is invalid
    """
    if not cylinders:
        return  # No cylinders to plot
        
    try:
        for (cx, cy, cr, ch) in cylinders:
            # Validation
            if cr <= 0 or ch <= 0:
                print(f"Warning: Invalid cylinder dimensions (r={cr}, h={ch}), skipping")
                continue
                
            # Cylinder side
            theta = np.linspace(0, 2 * np.pi, 32)
            z = np.linspace(0, ch, 10)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = cr * np.cos(theta_grid) + cx
            y_grid = cr * np.sin(theta_grid) + cy
            ax.plot_surface(x_grid, y_grid, z_grid, color=CYLINDER_COLOR, alpha=CYLINDER_ALPHA)
    
            # Bottom circle
            circle_x = cr * np.cos(theta) + cx
            circle_y = cr * np.sin(theta) + cy
            ax.plot(circle_x, circle_y, [0]*len(theta), color=CYLINDER_COLOR, alpha=CYLINDER_LINE_ALPHA)
            # Top circle
            ax.plot(circle_x, circle_y, [ch]*len(theta), color=CYLINDER_COLOR, alpha=CYLINDER_LINE_ALPHA)
            
    except Exception as e:
        raise ValueError(f"Error plotting cylinders: {e}")

def create_drone_animation(map_data, waypoints, duration=ANIMATION_DURATION, fps=FPS, 
                         view_angle=(30, 45), trail_length=20):
    """
    Create a smooth 3D animation of a drone moving along a given list of waypoints.
    The animation shows a dashed line for the planned path and a blue line smoothly
    following it to the destination.
    
    Args:
        map_data (dict): Must contain 'map_size', 'origin', 'destination', 'cylinders'.
        waypoints (list or np.ndarray): List/array of 3D coordinates [x, y, z].
        duration (float): Total animation duration in seconds (default: 5.0 seconds)
        fps (int): Frames per second for the animation
        view_angle (tuple): View angle (elevation, azimuth) for the 3D plot
        trail_length (int): Length of the trail behind the current position
        
    Returns:
        anim (FuncAnimation): The matplotlib animation object.
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If animation creation fails
    """
    # Validate input data
    if not map_data:
        raise ValueError("Map data is required")
    
    if 'map_size' not in map_data or 'origin' not in map_data or 'destination' not in map_data:
        raise ValueError("Map data is missing required keys")
    
    if not waypoints or len(waypoints) < 2:
        raise ValueError("At least two waypoints are required for animation")
    
    if duration <= 0:
        raise ValueError(f"Duration must be positive (got {duration})")
    
    if fps <= 0:
        raise ValueError(f"FPS must be positive (got {fps})")
        
    try:
        map_size = map_data['map_size']
        origin = map_data['origin']
        destination = map_data['destination']
        cylinders = map_data['cylinders'] if 'cylinders' in map_data else []
    
        # Convert waypoints to a NumPy array if needed
        waypoints = np.array(waypoints)
        
        # Validate waypoints have correct shape (N, 3)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError(f"Waypoints must have shape (N, 3) but got {waypoints.shape}")
        
        # Create interpolated points for smooth animation
        n_frames = int(fps * duration)
        if n_frames < 2:
            n_frames = 2  # Ensure at least 2 frames
        
        # Generate smooth path with distance-based interpolation
        cumulative_distances = [0]
        for i in range(1, len(waypoints)):
            d = np.linalg.norm(waypoints[i] - waypoints[i-1])
            cumulative_distances.append(cumulative_distances[-1] + d)
        
        # Normalize distances to [0, 1]
        if cumulative_distances[-1] > 0:
            waypoint_times = np.array(cumulative_distances) / cumulative_distances[-1]
        else:
            waypoint_times = np.linspace(0, 1, len(waypoints))
        
        # Generate interpolated points along the path
        interp_times = np.linspace(0, 1, n_frames)
        
        # Interpolate each dimension separately
        x_interp = np.interp(interp_times, waypoint_times, waypoints[:, 0])
        y_interp = np.interp(interp_times, waypoint_times, waypoints[:, 1])
        z_interp = np.interp(interp_times, waypoint_times, waypoints[:, 2])
        
        # Combine interpolated coordinates
        smooth_path = np.column_stack((x_interp, y_interp, z_interp))
    
        # Create figure with error handling
        try:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        except Exception as e:
            raise RuntimeError(f"Failed to create 3D plot: {e}")
    
        # Set the axes limits with padding
        buffer = map_size * 0.05  # 5% buffer
        ax.set_xlim(-buffer, map_size + buffer)
        ax.set_ylim(-buffer, map_size + buffer)
        
        # Calculate z-axis limit with padding
        if cylinders:
            try:
                tallest_cylinder = max([c[3] for c in cylinders])
            except (TypeError, IndexError):
                tallest_cylinder = 50
                print("Warning: Could not determine maximum cylinder height, using default")
        else:
            tallest_cylinder = 50
            
        # Find maximum z value in path
        max_z_path = np.max(smooth_path[:, 2]) if smooth_path.size > 0 else 0
        z_limit = max(tallest_cylinder, max_z_path) * 1.2  # 20% buffer
        ax.set_zlim(0, z_limit)
    
        # Plot ground
        X, Y = np.meshgrid(np.linspace(0, map_size, 50),
                          np.linspace(0, map_size, 50))
        Z = np.zeros_like(X)
        ax.plot_surface(X, Y, Z, color=GROUND_COLOR, alpha=0.3)
    
        # Plot cylinders
        plot_cylinders(ax, cylinders)
    
        # Plot origin (red) and destination (green)
        ax.scatter(origin[0], origin[1], origin[2], color=ORIGIN_COLOR, s=100, marker='o', label='Origin')
        ax.scatter(destination[0], destination[1], destination[2], color=DESTINATION_COLOR, s=100, marker='o', label='Destination')
        
        # Plot the planned path as a dashed line (more visible)
        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                '--', color=PLANNED_PATH_COLOR, linewidth=1.5, alpha=0.7, label='Planned Path')
        
        # Create a solid blue line that will follow the path
        path_line, = ax.plot([], [], [], '-', color=ACTUAL_PATH_COLOR, linewidth=2.5, alpha=0.8, label='Actual Path')
        
        # Add labels and legend
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Drone Flight Path Animation ({duration} seconds)")
        ax.legend()
        
        # Set view angle
        elev, azim = view_angle
        ax.view_init(elev=elev, azim=azim)
        
        # Define update function with error handling
        def update(frame):
            """
            Update function for animation - simple implementation to reduce visual artifacts
            """
            try:
                # Ensure frame is within bounds
                if frame >= len(smooth_path):
                    frame = len(smooth_path) - 1
                if frame < 0:
                    frame = 0
                    
                if frame > 0:
                    # Calculate the portion of the path to show
                    start_idx = max(0, frame - trail_length)
                    
                    # Update the blue line to show progress along the path
                    path_line.set_data(smooth_path[start_idx:frame+1, 0], smooth_path[start_idx:frame+1, 1])
                    path_line.set_3d_properties(smooth_path[start_idx:frame+1, 2])
                    
                return [path_line]
            except Exception as e:
                print(f"Warning: Animation update error: {e}")
                return [path_line]  # Return unchanged to avoid breaking animation
        
        # Calculate interval (in milliseconds) between frames
        interval = (duration * 1000) / n_frames
        
        # Use simple animation parameters for better compatibility
        try:
            anim = FuncAnimation(
                fig, 
                update,
                frames=n_frames,
                interval=interval,
                blit=False,  # Disable blitting for more reliable rendering
                repeat=True
            )
            
            # Store the animation in a figure attribute to prevent garbage collection
            fig._animation = anim
            
            return anim
            
        except Exception as e:
            plt.close(fig)  # Clean up figure resources
            raise RuntimeError(f"Failed to create animation: {e}")
            
    except Exception as e:
        plt.close()  # Clean up any open figures
        raise ValueError(f"Error creating animation: {e}")

def generate_flight_path(origin, destination, cylinders, altitude_buffer=10, waypoint_count=9):
    """
    Generate a realistic flight path from origin to destination.
    
    Args:
        origin: Starting point [x, y, z]
        destination: Ending point [x, y, z]
        cylinders: List of cylinders for obstacle height calculation
        altitude_buffer: Extra height above tallest obstacle
        waypoint_count: Number of waypoints to generate
        
    Returns:
        List of waypoints
    """
    try:
        # Calculate a reasonable flight altitude (above the tallest cylinder)
        tallest_cylinder = max([c[3] for c in cylinders]) if cylinders else 0
        flight_altitude = max(tallest_cylinder + altitude_buffer, 30)  # Minimum height of 30

        # Define waypoints for a smooth path
        # Start with takeoff
        waypoints = [
            [origin[0], origin[1], 0],                    # Start exactly at origin on ground
            [origin[0], origin[1], flight_altitude/2],    # Ascend halfway to cruise altitude
            [origin[0], origin[1], flight_altitude],      # Reach cruise altitude
        ]
        
        # Calculate intermediate waypoints
        segments = waypoint_count - 5  # Subtract takeoff (2) and landing (2) segments
        for i in range(1, segments+1):
            # Fraction of the way from origin to destination
            frac = i / (segments+1)
            
            # Create some altitude variation for a more interesting path
            altitude_variation = 5 * np.sin(frac * np.pi)  # Sinusoidal variation
            
            # Add waypoint
            waypoints.append([
                origin[0] + frac * (destination[0] - origin[0]),
                origin[1] + frac * (destination[1] - origin[1]),
                flight_altitude + altitude_variation
            ])
            
        # Add landing approach
        waypoints.extend([
            [destination[0], destination[1], flight_altitude/2],  # Begin descent
            [destination[0], destination[1], 0]                   # Land at destination
        ])
        
        return waypoints
        
    except Exception as e:
        print(f"Error generating flight path: {e}")
        # Fallback to a minimal path if generation fails
        return [
            [origin[0], origin[1], 0],
            [origin[0], origin[1], 30],
            [destination[0], destination[1], 30],
            [destination[0], destination[1], 0]
        ]

def main():
    """Main function with command-line argument handling"""
    parser = argparse.ArgumentParser(description='Animate a drone flight path using a 3D map')
    parser.add_argument('--map', default=DEFAULT_MAP_FILE, help=f'Path to the map pickle file (default: {DEFAULT_MAP_FILE})')
    parser.add_argument('--duration', type=float, default=ANIMATION_DURATION, help=f'Animation duration in seconds (default: {ANIMATION_DURATION})')
    parser.add_argument('--fps', type=int, default=FPS, help=f'Frames per second (default: {FPS})')
    parser.add_argument('--save', help='Save the animation to this file (e.g., animation.mp4)')
    parser.add_argument('--view-angle', type=str, default='30,45', help='View angle (elevation,azimuth) for 3D plot (default: 30,45)')
    parser.add_argument('--waypoints', type=int, default=9, help='Number of waypoints to generate (default: 9)')
    parser.add_argument('--altitude', type=float, default=10, help='Altitude buffer above tallest obstacle (default: 10)')
    
    args = parser.parse_args()

    try:
        # Parse view angle
        view_angle = (30, 45)  # Default
        if args.view_angle:
            try:
                angle_parts = args.view_angle.split(',')
                if len(angle_parts) == 2:
                    view_angle = (float(angle_parts[0]), float(angle_parts[1]))
            except ValueError:
                print(f"Warning: Invalid view angle format: {args.view_angle}. Using defaults.")
        
        # Load the map data
        print(f"Loading map from {args.map}...")
        start_time = time.time()
        data = load_map_data(args.map)
        loading_time = time.time() - start_time
        print(f"Map loaded in {loading_time:.2f} seconds")
        
        # Extract origin and destination
        origin = data['origin']
        destination = data['destination']
        cylinders = data['cylinders']
        
        # Print route information
        distance_2d = np.sqrt((origin[0]-destination[0])**2 + (origin[1]-destination[1])**2)
        print(f"Flight path from {origin[:2]} to {destination[:2]} (distance: {distance_2d:.2f} units)")
        
        # Generate flight path
        print(f"Generating flight path with {args.waypoints} waypoints...")
        waypoints = generate_flight_path(
            origin, destination, cylinders, 
            altitude_buffer=args.altitude,
            waypoint_count=args.waypoints
        )
        
        print(f"Creating animation ({args.duration} seconds at {args.fps} FPS)...")
        start_time = time.time()
        
        # Create animation
        animation = create_drone_animation(
            data, waypoints, 
            duration=args.duration,
            fps=args.fps,
            view_angle=view_angle
        )
        
        anim_time = time.time() - start_time
        print(f"Animation created in {anim_time:.2f} seconds")
        
        # Save animation if requested
        if args.save:
            print(f"Saving animation to {args.save}...")
            try:
                if args.save.lower().endswith('.gif'):
                    # Save as GIF
                    animation.save(args.save, writer='pillow', fps=args.fps)
                else:
                    # Save as MP4 or other format using ffmpeg
                    animation.save(args.save, writer='ffmpeg', fps=args.fps)
                print(f"Animation saved to {args.save}")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("Make sure you have the necessary libraries installed (pillow for GIF, ffmpeg for MP4)")
        
        # Display the animation
        print("Displaying animation. Close the window to exit.")
        plt.show()
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please make sure the map file exists at {args.map}")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except RuntimeError as e:
        print(f"Animation error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nAnimation interrupted by user")
        plt.close('all')  # Close all figures
        return 0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
