import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
import random
import math
import os
import sys
import time
import signal

# Import global configuration parameters
from config import *

class TimeoutException(Exception):
    """Exception raised when an operation times out."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutException("Operation timed out")

class MapGenerator:
    def __init__(self, map_size=MAP_SIZE, num_cylinders=NUM_CYLINDERS, 
                 min_height=MIN_CYLINDER_HEIGHT, max_height=MAX_CYLINDER_HEIGHT, 
                 min_distance_from_points=MIN_DISTANCE_FROM_POINTS, 
                 min_dest_distance=MIN_DESTINATION_DISTANCE, 
                 max_dest_distance=MAX_DESTINATION_DISTANCE):
        """
        Initialize the 3D map generator with parameters.
        Uses defaults from config.py if not specified.
        
        Args:
            map_size: Size of the square map (x and y dimensions)
            num_cylinders: Number of cylinders to generate
            min_height: Minimum height of cylinders
            max_height: Maximum height of cylinders
            min_distance_from_points: Minimum distance cylinders must be from origin/destination
            min_dest_distance: Minimum distance between origin and destination
            max_dest_distance: Maximum distance between origin and destination
            
        Raises:
            ValueError: If any parameters are invalid
        """
        # Validate input parameters
        if map_size <= 0:
            raise ValueError(f"Map size must be positive (got {map_size})")
        if num_cylinders < 0:
            raise ValueError(f"Number of cylinders cannot be negative (got {num_cylinders})")
        if min_height < 0:
            raise ValueError(f"Minimum height cannot be negative (got {min_height})")
        if max_height <= min_height:
            raise ValueError(f"Maximum height must be greater than minimum height ({max_height} <= {min_height})")
        if min_distance_from_points < 0:
            raise ValueError(f"Minimum distance from points cannot be negative (got {min_distance_from_points})")
        if min_dest_distance < 0:
            raise ValueError(f"Minimum destination distance cannot be negative (got {min_dest_distance})")
        if max_dest_distance <= min_dest_distance:
            raise ValueError(f"Maximum destination distance must be greater than minimum destination distance ({max_dest_distance} <= {min_dest_distance})")
        
        # Check if parameters make sense in relation to map size
        if max_dest_distance > map_size * 1.414:  # sqrt(2) * map_size is the max possible distance
            print(f"Warning: Maximum destination distance ({max_dest_distance}) exceeds the maximum possible distance in the map ({map_size * 1.414:.2f})")
            max_dest_distance = map_size * 1.414
            
        if min_distance_from_points * 2 + max_dest_distance > map_size * 1.414:
            print(f"Warning: Constraints may make cylinder placement difficult or impossible")
            
        self.map_size = map_size
        self.num_cylinders = num_cylinders
        self.min_height = min_height
        self.max_height = max_height
        self.min_distance_from_points = min_distance_from_points
        self.min_dest_distance = min_dest_distance
        self.max_dest_distance = max_dest_distance
        
        # Initialize origin at the bottom-left corner of the map
        self.origin = np.array([0, 0, 0])
        self.destination = None
        self.cylinders = []  # List to store cylinder data (x, y, radius, height)
        self.generation_timeout = 60  # Default timeout in seconds for map generation
        
    def distance_2d(self, point1, point2):
        """Calculate the 2D Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def generate_destination(self, max_attempts=100):
        """
        Generate a random destination point within distance constraints
        
        Args:
            max_attempts: Maximum number of attempts to make before giving up
            
        Raises:
            ValueError: If no valid destination could be found within constraints
            TimeoutException: If operation times out
        """
        attempts = 0
        start_time = time.time()
        
        try:
            # Set a timeout for this operation
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.generation_timeout)
            
            while attempts < max_attempts:
                attempts += 1
                
                # Since origin is at corner, we need to adjust how we generate the destination
                # Use a random position within the map that meets our distance criteria
                x = random.uniform(self.min_dest_distance / 2, self.map_size)
                y = random.uniform(self.min_dest_distance / 2, self.map_size)
                
                # Calculate distance from origin
                distance = self.distance_2d([x, y], self.origin[:2])
                
                # Check if distance criteria is met
                if self.min_dest_distance <= distance <= self.max_dest_distance:
                    self.destination = np.array([x, y, 0])
                    signal.alarm(0)  # Cancel the timeout
                    return
                
                # If many attempts fail, gradually relax constraints
                if attempts > max_attempts * 0.8:
                    relaxed_min = self.min_dest_distance * 0.9
                    relaxed_max = min(self.max_dest_distance * 1.1, self.map_size * 1.414)
                    if relaxed_min <= distance <= relaxed_max:
                        print(f"Warning: Relaxed distance constraints to find valid destination")
                        self.destination = np.array([x, y, 0])
                        signal.alarm(0)  # Cancel the timeout
                        return
            
            # If we get here, we couldn't find a valid destination
            raise ValueError(f"Could not find a valid destination within constraints after {max_attempts} attempts. Try adjusting distance parameters.")
            
        except TimeoutException:
            raise TimeoutException("Destination generation timed out. Try adjusting the distance parameters.")
        finally:
            signal.alarm(0)  # Cancel the timeout in case of exception
    
    def generate_cylinders(self, max_attempts_per_cylinder=50):
        """
        Generate random cylinders with varying heights
        
        Args:
            max_attempts_per_cylinder: Maximum attempts to place each cylinder
            
        Raises:
            ValueError: If unable to place all cylinders
            TimeoutException: If operation times out
        """
        if self.destination is None:
            raise ValueError("Destination must be generated before cylinders")
        
        self.cylinders = []  # Reset cylinders list
        total_attempts = 0
        max_total_attempts = self.num_cylinders * max_attempts_per_cylinder
        cylinders_placed = 0
        start_time = time.time()
        
        try:
            # Set a timeout for this operation
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.generation_timeout)
            
            while cylinders_placed < self.num_cylinders and total_attempts < max_total_attempts:
                total_attempts += 1
                
                # Generate random position and properties
                x = random.uniform(0, self.map_size)
                y = random.uniform(0, self.map_size)
                radius = random.uniform(MIN_CYLINDER_RADIUS, MAX_CYLINDER_RADIUS)
                height = random.uniform(self.min_height, self.max_height)
                
                # Check if cylinder is too close to origin or destination
                if (self.distance_2d([x, y], self.origin[:2]) < self.min_distance_from_points + radius or
                    self.distance_2d([x, y], self.destination[:2]) < self.min_distance_from_points + radius):
                    continue
                
                # Check if cylinder overlaps with existing cylinders
                overlap = False
                for cx, cy, cr, _ in self.cylinders:
                    if self.distance_2d([x, y], [cx, cy]) < radius + cr:
                        overlap = True
                        break
                
                if not overlap:
                    self.cylinders.append((x, y, radius, height))
                    cylinders_placed += 1
                    
                # If we're struggling to place cylinders, gradually relax constraints
                if total_attempts > max_total_attempts * 0.8 and cylinders_placed < self.num_cylinders * 0.7:
                    reduced_radius = MIN_CYLINDER_RADIUS  # Use minimum radius
                    x = random.uniform(0, self.map_size)
                    y = random.uniform(0, self.map_size)
                    height = random.uniform(self.min_height, self.max_height)
                    
                    # Check with reduced constraints
                    if (self.distance_2d([x, y], self.origin[:2]) < self.min_distance_from_points * 0.8 + reduced_radius or
                        self.distance_2d([x, y], self.destination[:2]) < self.min_distance_from_points * 0.8 + reduced_radius):
                        continue
                    
                    # Check for overlaps with relaxed constraints
                    overlap = False
                    for cx, cy, cr, _ in self.cylinders:
                        if self.distance_2d([x, y], [cx, cy]) < reduced_radius + cr * 0.9:
                            overlap = True
                            break
                    
                    if not overlap:
                        self.cylinders.append((x, y, reduced_radius, height))
                        cylinders_placed += 1
            
            signal.alarm(0)  # Cancel the timeout
            
            # If we couldn't place enough cylinders, warn the user
            if cylinders_placed < self.num_cylinders:
                percentage = (cylinders_placed / self.num_cylinders) * 100
                if percentage < 50:
                    raise ValueError(f"Only placed {cylinders_placed}/{self.num_cylinders} cylinders ({percentage:.1f}%). Try reducing the number of cylinders or adjusting placement constraints.")
                else:
                    print(f"Warning: Only placed {cylinders_placed}/{self.num_cylinders} cylinders ({percentage:.1f}%)")
            
            # Verify we have at least some cylinders
            if not self.cylinders:
                raise ValueError(ERROR_NO_CYLINDERS)
                
        except TimeoutException:
            if self.cylinders:
                print(f"Warning: Cylinder generation timed out after placing {len(self.cylinders)}/{self.num_cylinders} cylinders")
            else:
                raise TimeoutException(ERROR_MAP_GENERATION_TIMEOUT)
        finally:
            signal.alarm(0)  # Cancel the timeout in case of exception
    
    def generate_map(self):
        """
        Generate the complete 3D map
        
        Raises:
            ValueError: If map cannot be generated with current parameters
            TimeoutException: If operation times out
        """
        try:
            self.generate_destination()
            self.generate_cylinders()
            return True
        except Exception as e:
            # Clean up partial map generation
            self.destination = None
            self.cylinders = []
            raise e
    
    def save_map(self, filename=DEFAULT_MAP_FILE):
        """
        Save the map data to a file
        
        Args:
            filename: Path to save the map data (pickle) file
            
        Returns:
            filename: The path to the saved file
            
        Raises:
            ValueError: If map data is invalid
            IOError: If file cannot be written
        """
        # Validation
        if self.destination is None:
            raise ValueError("Cannot save map: destination not generated")
        if not self.cylinders:
            raise ValueError("Cannot save map: no cylinders generated")
            
        map_data = {
            "map_size": self.map_size,
            "origin": self.origin,
            "destination": self.destination,
            "cylinders": self.cylinders
        }
        
        try:
            # Ensure the directory exists
            directory = os.path.dirname(os.path.abspath(filename))
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(filename, 'wb') as f:
                pickle.dump(map_data, f)
            
            return filename
        except (IOError, OSError) as e:
            raise IOError(f"Failed to save map to {filename}: {str(e)}")
    
    def visualize_map(self, show=True, save_fig=True, filename=DEFAULT_IMG_FILE):
        """
        Visualize the 3D map with matplotlib
        
        Args:
            show: Whether to display the visualization
            save_fig: Whether to save the visualization to a file
            filename: Path to save the visualization image
            
        Returns:
            fig: The matplotlib figure object
            
        Raises:
            ValueError: If map data is invalid
            IOError: If figure cannot be saved
        """
        # Validation
        if self.destination is None:
            raise ValueError("Cannot visualize map: destination not generated")
        if not self.cylinders:
            raise ValueError("Cannot visualize map: no cylinders generated")
            
        try:
            # Create the figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Set the limits of the plot
            ax.set_xlim(0, self.map_size)
            ax.set_ylim(0, self.map_size)
            
            # Calculate the maximum height for the z-axis
            max_height = max([h for _, _, _, h in self.cylinders]) if self.cylinders else self.max_height
            buffer = max(10, max_height * 0.2)  # Add some buffer space
            ax.set_zlim(0, max_height + buffer)
            
            # Create a base map (ground)
            x = np.linspace(0, self.map_size, 100)
            y = np.linspace(0, self.map_size, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            
            # Plot the ground as a surface
            ax.plot_surface(X, Y, Z, color=GROUND_COLOR, alpha=0.3, zorder=1)
            
            # Plot cylinders
            for x, y, radius, height in self.cylinders:
                # Create a circle in the x-y plane
                theta = np.linspace(0, 2 * np.pi, 32)
                circle_x = radius * np.cos(theta) + x
                circle_y = radius * np.sin(theta) + y
                
                # Create cylinder
                # Bottom circle
                ax.plot(circle_x, circle_y, np.zeros_like(theta), color=CYLINDER_COLOR, alpha=CYLINDER_LINE_ALPHA)
                # Top circle
                ax.plot(circle_x, circle_y, np.ones_like(theta) * height, color=CYLINDER_COLOR, alpha=CYLINDER_LINE_ALPHA)
                
                # Connect bottom and top circles with lines
                for i in range(0, len(theta), 4):
                    ax.plot([circle_x[i], circle_x[i]], [circle_y[i], circle_y[i]], 
                            [0, height], color=CYLINDER_COLOR, alpha=CYLINDER_LINE_ALPHA)
                
                # Create cylinder surface
                z = np.linspace(0, height, 10)
                theta_grid, z_grid = np.meshgrid(theta, z)
                x_grid = radius * np.cos(theta_grid) + x
                y_grid = radius * np.sin(theta_grid) + y
                
                ax.plot_surface(x_grid, y_grid, z_grid, color=CYLINDER_COLOR, alpha=CYLINDER_ALPHA)
            
            # Plot origin point (red) and destination point (green)
            ax.scatter(self.origin[0], self.origin[1], self.origin[2], color=ORIGIN_COLOR, s=100, marker='o', label='Origin')
            ax.scatter(self.destination[0], self.destination[1], self.destination[2], color=DESTINATION_COLOR, s=100, marker='o', label='Destination')
            
            # Add a direct line from origin to destination
            ax.plot([self.origin[0], self.destination[0]], 
                    [self.origin[1], self.destination[1]], 
                    [self.origin[2], self.destination[2]], '--', color=PLANNED_PATH_COLOR, alpha=0.5)
            
            # Calculate the distance for the title
            distance = self.distance_2d(self.origin[:2], self.destination[:2])
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D Map with {len(self.cylinders)} Cylinders\nDistance: {distance:.2f} units')
            ax.legend()
            
            # Adjust view angle
            ax.view_init(elev=30, azim=45)
            
            # Save the figure if requested
            if save_fig:
                try:
                    # Ensure the directory exists
                    directory = os.path.dirname(os.path.abspath(filename))
                    if directory and not os.path.exists(directory):
                        os.makedirs(directory)
                    
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"Figure saved as {filename}")
                except (IOError, OSError) as e:
                    print(f"Warning: Could not save figure to {filename}: {str(e)}")
                
            # Show the figure if requested
            if show:
                plt.show()
            
            # Close the figure to free resources
            plt.close(fig)
            return fig
            
        except Exception as e:
            plt.close()  # Make sure to close the figure even if something fails
            raise e

def generate_and_save_map(map_size=MAP_SIZE, num_cylinders=NUM_CYLINDERS, show_visualization=True,
                     output_file=DEFAULT_MAP_FILE, image_file=DEFAULT_IMG_FILE, timeout=60):
    """
    Generate, save, and optionally visualize a 3D map using global configuration
    
    Args:
        map_size: Size of the square map
        num_cylinders: Number of cylinders to generate
        show_visualization: Whether to display the visualization
        output_file: File to save the map data to
        image_file: File to save the visualization to
        timeout: Timeout in seconds for map generation
        
    Returns:
        tuple: (map_file, map_gen) - The path to the saved map file and the MapGenerator instance
        
    Raises:
        ValueError: If map generation fails due to parameter constraints
        TimeoutException: If map generation times out
    """
    try:
        # Create a map generator with the specified parameters
        map_gen = MapGenerator(
            map_size=map_size,
            num_cylinders=num_cylinders
        )
        
        # Set the timeout for map generation
        map_gen.generation_timeout = timeout
        
        # Generate the map
        start_time = time.time()
        map_gen.generate_map()
        generation_time = time.time() - start_time
        
        # Save the map
        map_file = map_gen.save_map(output_file)
        
        # Print information about the generated map
        print(f"Map generated in {generation_time:.2f} seconds")
        print(f"Map contains {len(map_gen.cylinders)} cylinders")
        print(f"Origin: {map_gen.origin[:2]}")
        print(f"Destination: {map_gen.destination[:2]}")
        
        distance = map_gen.distance_2d(map_gen.origin[:2], map_gen.destination[:2])
        print(f"Distance between origin and destination: {distance:.2f} units")
        
        # Print statistics about cylinders
        if map_gen.cylinders:
            heights = [h for _, _, _, h in map_gen.cylinders]
            radii = [r for _, _, r, _ in map_gen.cylinders]
            print(f"Cylinder height range: {min(heights):.2f} to {max(heights):.2f} units")
            print(f"Cylinder radius range: {min(radii):.2f} to {max(radii):.2f} units")
            
        print(f"Map data saved to {map_file}")
        
        # Visualize the map if requested
        if show_visualization:
            map_gen.visualize_map(filename=image_file)
        
        return map_file, map_gen
        
    except TimeoutException as e:
        print(f"Error: {str(e)}")
        print("Try adjusting the parameters or increasing the timeout.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("Try adjusting the parameters.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a 3D map for drone flight path planning")
    parser.add_argument("--size", type=int, default=MAP_SIZE, help=f"Size of the map (default: {MAP_SIZE})")
    parser.add_argument("--cylinders", type=int, default=NUM_CYLINDERS, help=f"Number of cylinders to generate (default: {NUM_CYLINDERS})")
    parser.add_argument("--min-height", type=float, default=MIN_CYLINDER_HEIGHT, help=f"Minimum height of cylinders (default: {MIN_CYLINDER_HEIGHT})")
    parser.add_argument("--max-height", type=float, default=MAX_CYLINDER_HEIGHT, help=f"Maximum height of cylinders (default: {MAX_CYLINDER_HEIGHT})")
    parser.add_argument("--min-distance", type=float, default=MIN_DESTINATION_DISTANCE, 
                        help=f"Minimum distance between origin and destination (default: {MIN_DESTINATION_DISTANCE})")
    parser.add_argument("--output", "-o", default=DEFAULT_MAP_FILE, help=f"Output file for map data (default: {DEFAULT_MAP_FILE})")
    parser.add_argument("--image", "-i", default=DEFAULT_IMG_FILE, help=f"Output file for visualization (default: {DEFAULT_IMG_FILE})")
    parser.add_argument("--no-display", action="store_true", help="Do not display the visualization")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds for map generation (default: 60)")
    
    args = parser.parse_args()
    
    # Update global configuration parameters from command line arguments
    if args.min_height != MIN_CYLINDER_HEIGHT or args.max_height != MAX_CYLINDER_HEIGHT:
        MIN_CYLINDER_HEIGHT = args.min_height
        MAX_CYLINDER_HEIGHT = args.max_height
        print(f"Using custom cylinder height range: {MIN_CYLINDER_HEIGHT} to {MAX_CYLINDER_HEIGHT}")
    
    if args.min_distance != MIN_DESTINATION_DISTANCE:
        MIN_DESTINATION_DISTANCE = args.min_distance
        print(f"Using custom minimum destination distance: {MIN_DESTINATION_DISTANCE}")
    
    # Generate the map
    map_file, map_gen = generate_and_save_map(
        map_size=args.size,
        num_cylinders=args.cylinders,
        show_visualization=not args.no_display,
        output_file=args.output,
        image_file=args.image,
        timeout=args.timeout
    )