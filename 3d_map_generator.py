import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle
import random
import math

class MapGenerator:
    def __init__(self, map_size=100, num_cylinders=30, min_height=10, max_height=50, 
                 min_distance_from_points=10, min_dest_distance=40, max_dest_distance=120):
        """
        Initialize the 3D map generator with parameters.
        
        Args:
            map_size: Size of the square map (x and y dimensions)
            num_cylinders: Number of cylinders to generate
            min_height: Minimum height of cylinders
            max_height: Maximum height of cylinders
            min_distance_from_points: Minimum distance cylinders must be from origin/destination
            min_dest_distance: Minimum distance between origin and destination
            max_dest_distance: Maximum distance between origin and destination
        """
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
        
    def distance_2d(self, point1, point2):
        """Calculate the 2D Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def generate_destination(self):
        """Generate a random destination point within distance constraints"""
        while True:
            # Since origin is at corner, we need to adjust how we generate the destination
            # Use a random position within the map that meets our distance criteria
            x = random.uniform(self.min_dest_distance, self.map_size)
            y = random.uniform(self.min_dest_distance, self.map_size)
            
            # Calculate distance from origin
            distance = self.distance_2d([x, y], self.origin[:2])
            
            # Check if distance criteria is met
            if self.min_dest_distance <= distance <= self.max_dest_distance:
                self.destination = np.array([x, y, 0])
                break
    
    def generate_cylinders(self):
        """Generate random cylinders with varying heights"""
        attempts = 0
        max_attempts = self.num_cylinders * 10  # Limit attempts to avoid infinite loops
        
        while len(self.cylinders) < self.num_cylinders and attempts < max_attempts:
            attempts += 1
            
            # Generate random position and properties
            x = random.uniform(0, self.map_size)
            y = random.uniform(0, self.map_size)
            radius = random.uniform(2, 5)
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
    
    def generate_map(self):
        """Generate the complete 3D map"""
        self.generate_destination()
        self.generate_cylinders()
    
    def save_map(self, filename="3d_map.pkl"):
        """Save the map data to a file"""
        map_data = {
            "map_size": self.map_size,
            "origin": self.origin,
            "destination": self.destination,
            "cylinders": self.cylinders
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(map_data, f)
        
        return filename
    
    def visualize_map(self, show=True, save_fig=True, filename="3d_map.png"):
        """Visualize the 3D map with matplotlib"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set the limits of the plot
        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)
        ax.set_zlim(0, self.max_height + 10)
        
        # Create a base map (ground)
        x = np.linspace(0, self.map_size, 100)
        y = np.linspace(0, self.map_size, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Plot the ground as a surface
        ax.plot_surface(X, Y, Z, color='lightgray', alpha=0.3, zorder=1)
        
        # Plot cylinders
        for x, y, radius, height in self.cylinders:
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
        ax.scatter(self.origin[0], self.origin[1], self.origin[2], color='red', s=100, marker='o', label='Origin')
        ax.scatter(self.destination[0], self.destination[1], self.destination[2], color='green', s=100, marker='o', label='Destination')
        
        # Add a direct line from origin to destination
        ax.plot([self.origin[0], self.destination[0]], 
                [self.origin[1], self.destination[1]], 
                [self.origin[2], self.destination[2]], 'k--', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Map with Cylinders')
        ax.legend()
        
        # Adjust view angle
        ax.view_init(elev=30, azim=45)
        
        if save_fig:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {filename}")
            
        if show:
            plt.show()
        
        plt.close()
        return fig

def generate_and_save_map(map_size=100, num_cylinders=30, show_visualization=True):
    """Generate, save, and optionally visualize a 3D map"""
    map_gen = MapGenerator(
        map_size=map_size,
        num_cylinders=num_cylinders,
        min_height=10,
        max_height=50,
        min_distance_from_points=10,
        min_dest_distance=40,
        max_dest_distance=90  # Increased max distance since origin is at corner now
    )
    
    map_gen.generate_map()
    map_file = map_gen.save_map()
    
    print(f"Map generated with {len(map_gen.cylinders)} cylinders")
    print(f"Origin: {map_gen.origin[:2]}")
    print(f"Destination: {map_gen.destination[:2]}")
    print(f"Distance between origin and destination: {map_gen.distance_2d(map_gen.origin[:2], map_gen.destination[:2]):.2f}")
    print(f"Map data saved to {map_file}")
    
    if show_visualization:
        map_gen.visualize_map()
    
    return map_file, map_gen

if __name__ == "__main__":
    map_file, map_gen = generate_and_save_map(show_visualization=True)