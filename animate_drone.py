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
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)

        # Bottom circle
        circle_x = cr * np.cos(theta) + cx
        circle_y = cr * np.sin(theta) + cy
        ax.plot(circle_x, circle_y, [0]*len(theta), 'b-', alpha=0.6)
        # Top circle
        ax.plot(circle_x, circle_y, [ch]*len(theta), 'b-', alpha=0.6)

def create_drone_animation(map_data, waypoints):
    """
    Create a 3D animation of a drone moving along a given list of waypoints.
    
    Args:
        map_data (dict): Must contain 'map_size', 'origin', 'destination', 'cylinders'.
        waypoints (list or np.ndarray): List/array of 3D coordinates [x, y, z].
    Returns:
        anim (FuncAnimation): The matplotlib animation object.
    """
    map_size = map_data['map_size']
    origin = map_data['origin']        # [x0, y0, z0]
    destination = map_data['destination']  # [xd, yd, zd]
    cylinders = map_data['cylinders']

    # Convert waypoints to a NumPy array if needed
    waypoints = np.array(waypoints)
    n_frames = len(waypoints)

    # Prepare the figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set the axes limits
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    # Ensure the z-limit can accommodate the tallest cylinder or a default of 50
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

    # Create a scatter point for the drone (starting at the first waypoint)
    drone_scatter = ax.scatter(waypoints[0, 0], waypoints[0, 1], waypoints[0, 2],
                               color='blue', s=50, marker='^', label='Drone')

    # Add labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Drone Animation Using Fixed Waypoints")
    ax.legend()

    def update(frame):
        """
        Update function for FuncAnimation. Moves the drone scatter to the next waypoint.
        The drone will jump from one waypoint to the next at each frame.
        """
        x, y, z = waypoints[frame]
        # For Matplotlib 3D scatter, we update with offsets
        drone_scatter._offsets3d = (np.array([x]), np.array([y]), np.array([z]))
        return (drone_scatter,)

    # Create the animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=600, blit=False)
    # 'interval=600' means 600 milliseconds per frame, adjust as desired

    return anim

if __name__ == "__main__":
    # -----------------------------------------------------------
    # 1) LOAD THE MAP
    # -----------------------------------------------------------
    data = load_map_data("3d_map.pkl")

    # -----------------------------------------------------------
    # 2) DEFINE WAYPOINTS
    #    - Replace these with your own 3D path
    # -----------------------------------------------------------
    waypoints = [
        [data['origin'][0], data['origin'][1], 10],    # Start slightly above the origin
        [10, 5, 15],
        [25, 20, 30],
        [50, 50, 25],
        [70, 70, 10],
        [data['destination'][0], data['destination'][1], 5]  # Arrive near the destination
    ]

    # -----------------------------------------------------------
    # 3) CREATE AND SHOW THE ANIMATION
    # -----------------------------------------------------------
    animation = create_drone_animation(data, waypoints)
    plt.show()
    # Optional: you can save the animation as a GIF or MP4
    # e.g. animation.save('drone_waypoints.gif', writer='imagemagick', fps=1)
