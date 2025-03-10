# **Programming Assignment: Optimal Drone Path Planning**  

## **Objective**  
You are to develop a Python program that calculates and outputs the most optimal flight path for a First-Person View (FPV) drone navigating a randomized 3D environment with obstacles. The drone must efficiently travel from its origin (takeoff point) to its destination (landing point) while avoiding obstacles and adhering to flight constraints. The goal is to write a program that provides the 3D waypoints of the shortest flight path from the origin to destination.

The program must:
- Use the provided `3d_map_generator.py`, to generate a randomized 3D map with vertical cylindrical obstacles (trees).
- Compute an optimal 3D path from the origin to the destination.
- Ensure compliance with flight constraints to prevent collisions and maintain safe altitude.
- Output 3D waypoints for the flight path.
- Use the provided `animate_drone.py` script to visualize the drone's movement.

---

## **Environment Details**  
The drone will operate in a randomly generated 3D space created using the provided `3d_map_generator.py` script. This environment contains:
- Trees as vertical cylindrical obstacles with randomized positions and radii
- A destination point that is randomly placed on the XY plane (z = 0)
- The origin point is always fixed at (0, 0, 0).

---

## **Requirements**  
Your Python program must accomplish the following tasks:

### 1. Generate a 3D Environment
- Run `3d_map_generator.py` to generate a randomized map.
- Extract the positions and sizes of obstacles from the generated map.
- Extract the randomized destination point (x, y, 0).

### **2. Compute Optimal Flight Path**
- Implement a path planning algorithm to determine the shortest, obstacle-free path from the origin to the destination.
- The path should consist of discrete 3D waypoints (x, y, z) that the drone can follow.

### **3. Flight Constraints**
Your drone's path must adhere to the following constraints:
  - The drone must not come within 1 unit of any obstacle (tree cylinder). This includes both horizontal (XY plane) and vertical (Z-axis) clearances.
  - The drone cannot fly below 10 unit above the ground at any point during flight.
  - Exception: The drone may descend below 10 unit only if it is within 15 unit of the origin (for takeoff) or the destination (for landing).
  - The path should avoid unnecessary sharp turns or sudden altitude changes

### **4. Output 3D Waypoints**
- The computed flight path should be represented as a list of waypoints in the format:
  ```python
  waypoints = [(x1, y1, z1), (x2, y2, z2), ..., (xn, yn, zn)]
  ```
- Waypoints should be evenly spaced to ensure a smooth flight.

### **5. Visualize the Drone Flight Path**
- Use `animate_drone.py` to simulate the flight
- The animation should clearly show the drone:
  - Taking off from the origin
  - Navigating around obstacles while maintaining flight constraints.
  - Landing at the randomized destination


## **Example Output**
After running your program, it should print the computed waypoints as follows:

```python
Optimal flight path waypoints:
[
  (0, 0, 0),
  (5, 5, 15),
  (10, 15, 20),
  (20, 25, 20),
  (30, 30, 15),
  (destination_x, destination_y, 0)
]
```
Then, `animate_drone.py` should visualize the drone moving smoothly from origin to destination while avoiding obstacles.

Make sure that the first waypoint is the origin and the last waypoint is the destination coordinate.

Additionally:
- Include a short **SOLUTION.md** describing:
    - How your algorithm works.
    - Any external libraries used.
    - How to run the program.
- Answer the following questions:
    - Are there any parts of the program that are not fully functional?
    - What was the hardest part of this project?
    - What improvments could the program have to reflect real world scenarios?

Good luck!