import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def load_map_data(pkl_file='3d_map.pkl'):
    """
    Load map data (map_size, origin, destination, cylinders) from a pickle file.
    Returns a dict with keys: "map_size", "origin", "destination", "cylinders".
    """
    with open(pkl_file, 'rb') as f:
        map_data = pickle.load(f)
    return map_data

def plot_cylinders(ax, cylinders):
    """
    Plot cylinders in the 3D axis.
    Each cylinder is given by (x_center, y_center, radius, height).
    """
    for (cx, cy, cr, ch) in cylinders:
        # Cylinder side
        theta = np.linspace(0, 2 * np.pi, 32)
        z = np.linspace(0, ch, 10)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = cr * np.cos(theta_grid) + cx
        y_grid = cr * np.sin(theta_grid) + cy
        ax.plot_surface(x_grid, y_grid, z_grid, color='grey', alpha=0.3)

        # Bottom circle
        circle_x = cr * np.cos(theta) + cx
        circle_y = cr * np.sin(theta) + cy
        ax.plot(circle_x, circle_y, [0]*len(theta), color='grey', alpha=0.4)
        # Top circle
        ax.plot(circle_x, circle_y, [ch]*len(theta), color='grey', alpha=0.4)

def create_drone_animation(map_data, waypoints, duration=5.0):
    """
    Create a smooth 3D animation of a drone moving along a given list of waypoints.
    The animation shows a dashed line for the planned path and a blue line smoothly
    following it to the destination.
    
    Args:
        map_data (dict): Must contain 'map_size', 'origin', 'destination', 'cylinders'.
        waypoints (list or np.ndarray): List/array of 3D coordinates [x, y, z].
        duration (float): Total animation duration in seconds (default: 5.0 seconds)
    Returns:
        anim (FuncAnimation): The matplotlib animation object.
    """
    map_size = map_data['map_size']
    origin = map_data['origin']
    destination = map_data['destination']
    cylinders = map_data['cylinders']

    # Convert waypoints to a NumPy array if needed
    waypoints = np.array(waypoints)
    
    # Create interpolated points for smooth animation (20 FPS)
    n_frames = int(20 * duration)
    
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

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set the axes limits
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    tallest_cylinder = max([c[3] for c in cylinders]) if cylinders else 50
    ax.set_zlim(0, max(tallest_cylinder, 50))

    # Plot ground
    X, Y = np.meshgrid(np.linspace(0, map_size, 50),
                      np.linspace(0, map_size, 50))
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.3)

    # Plot cylinders
    plot_cylinders(ax, cylinders)

    # Plot origin (red) and destination (green)
    ax.scatter(origin[0], origin[1], origin[2], color='red', s=100, marker='o', label='Origin')
    ax.scatter(destination[0], destination[1], destination[2], color='green', s=100, marker='o', label='Destination')
    
    # Plot the planned path as a dashed line (more visible)
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
            'k--', linewidth=1.5, alpha=0.7, label='Planned Path')
    
    # Create a solid blue line that will follow the path
    path_line, = ax.plot([], [], [], 'b-', linewidth=2.5, alpha=0.8, label='Actual Path')
    
    # Add labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Drone Flight Path Animation ({duration} seconds)")
    ax.legend()
    
    def update(frame):
        """
        Update function for animation - simple implementation to reduce visual artifacts
        """
        if frame > 0:
            # Update the blue line to show progress along the path
            path_line.set_data(smooth_path[:frame+1, 0], smooth_path[:frame+1, 1])
            path_line.set_3d_properties(smooth_path[:frame+1, 2])
            
        return [path_line]
    
    # Create animation with a simple interval calculation
    interval = (duration * 1000) / n_frames
    
    # Use simple animation parameters for better compatibility
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

if __name__ == "__main__":
    # -----------------------------------------------------------
    # 1) LOAD THE MAP
    # -----------------------------------------------------------
    data = load_map_data("3d_map.pkl")

    # -----------------------------------------------------------
    # 2) DEFINE WAYPOINTS
    #    - This creates a more interesting and realistic path
    # -----------------------------------------------------------
    # Extract origin and destination
    origin = data['origin']
    destination = data['destination']
    
    # Calculate a reasonable flight altitude (above the tallest cylinder)
    cylinders = data['cylinders']
    tallest_cylinder = max([c[3] for c in cylinders]) if cylinders else 0
    flight_altitude = max(tallest_cylinder + 10, 30)  # At least 10 units above tallest obstacle
    
    # Define more waypoints for a smoother path
    # Start by ascending from origin
    waypoints = [
        [origin[0], origin[1], 0],                    # Start exactly at origin on ground
        [origin[0], origin[1], flight_altitude/2],     # Ascend halfway to cruise altitude
        [origin[0], origin[1], flight_altitude],      # Reach cruise altitude
        
        # Create a path that moves toward destination with some altitude variations
        [origin[0] + 0.2*(destination[0]-origin[0]), 
         origin[1] + 0.2*(destination[1]-origin[1]), 
         flight_altitude + 5],
         
        [origin[0] + 0.4*(destination[0]-origin[0]), 
         origin[1] + 0.4*(destination[1]-origin[1]), 
         flight_altitude - 3],
         
        [origin[0] + 0.6*(destination[0]-origin[0]), 
         origin[1] + 0.6*(destination[1]-origin[1]), 
         flight_altitude],
         
        [origin[0] + 0.8*(destination[0]-origin[0]), 
         origin[1] + 0.8*(destination[1]-origin[1]), 
         flight_altitude - 5],
        
        # Start descending toward destination
        [destination[0], destination[1], flight_altitude/2],  # Begin descent
        [destination[0], destination[1], 0]                   # Land at destination
    ]

    # -----------------------------------------------------------
    # 3) CREATE AND SHOW THE ANIMATION
    # -----------------------------------------------------------
    # Create the animation with a 5-second duration
    animation = create_drone_animation(data, waypoints, duration=5.0)
    
    # Display the animation
    plt.show()
    
    # Optional: Save the animation as a GIF or MP4
    # animation.save('drone_flight_path.gif', writer='pillow', fps=60)
    # If you have ffmpeg: animation.save('drone_flight_path.mp4', writer='ffmpeg', fps=60)
