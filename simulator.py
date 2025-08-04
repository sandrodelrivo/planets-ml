import math
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Object:
    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray, name: str = ""):
        """
        Initialize an object with mass, position, and velocity.
        
        Args:
            mass: Mass of the object (kg)
            position: 2D position vector [x, y] (m)
            velocity: 2D velocity vector [vx, vy] (m/s)
            name: Optional name for the object
        """
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.force = np.zeros(2)
        self.name = name
        
    def __repr__(self):
        return f"Object(name='{self.name}', mass={self.mass}, pos={self.position}, vel={self.velocity})"


class PhysicsSimulator:
    def __init__(self, gravitational_constant: float = 6.67430e-11):
        """
        Initialize the physics simulator.
        
        Args:
            gravitational_constant: G constant (m�/kg�s�)
        """
        self.G = gravitational_constant
        self.objects: List[Object] = []
        self.time = 0.0
        
        # Visualization properties
        self.visualization_enabled = False
        self.fig = None
        self.ax = None
        self.trails = {}
        self.trail_length = 100
        self.fixed_limits = None
        
    def add_object(self, mass: float, position: List[float], velocity: List[float], name: str = "") -> Object:
        """
        Add an object to the simulation.
        
        Args:
            mass: Mass of the object (kg)
            position: [x, y] position (m)
            velocity: [vx, vy] velocity (m/s)
            name: Optional name for the object
            
        Returns:
            The created Object
        """
        obj = Object(mass, position, velocity, name)
        self.objects.append(obj)
        return obj
        
    def calculate_gravitational_force(self, obj1: Object, obj2: Object) -> np.ndarray:
        """
        Calculate gravitational force that obj2 exerts on obj1.
        
        Args:
            obj1: Object experiencing the force
            obj2: Object exerting the force
            
        Returns:
            Force vector [fx, fy] (N)
        """
        # Vector from obj1 to obj2
        r_vec = obj2.position - obj1.position
        r_magnitude = np.linalg.norm(r_vec)
        
        # Avoid division by zero for overlapping objects
        if r_magnitude == 0:
            return np.zeros(2)
            
        # Unit vector pointing from obj1 to obj2
        r_unit = r_vec / r_magnitude
        
        # Gravitational force magnitude: F = G * m1 * m2 / r�
        force_magnitude = self.G * obj1.mass * obj2.mass / (r_magnitude ** 2)
        
        # Force vector
        force_vector = force_magnitude * r_unit
        
        return force_vector
        
    def calculate_total_forces(self):
        """Calculate total gravitational forces on all objects."""
        # Reset forces
        for obj in self.objects:
            obj.force = np.zeros(2)
            
        # Calculate pairwise forces
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i != j:  # Don't calculate force of object on itself
                    force = self.calculate_gravitational_force(obj1, obj2)
                    obj1.force += force
                    
    def get_force_on_object(self, obj: Object) -> np.ndarray:
        """
        Get the current total force acting on a specific object.
        
        Args:
            obj: The object to get force for
            
        Returns:
            Force vector [fx, fy] (N)
        """
        return obj.force.copy()
        
    def get_force_on_object_by_name(self, name: str) -> np.ndarray:
        """
        Get the current total force acting on an object by name.
        
        Args:
            name: Name of the object
            
        Returns:
            Force vector [fx, fy] (N)
        """
        for obj in self.objects:
            if obj.name == name:
                return obj.force.copy()
        raise ValueError(f"Object with name '{name}' not found")
        
    def update_physics(self, dt: float):
        """
        Update the simulation by one time step using Euler integration.
        
        Args:
            dt: Time step (s)
        """
        # Calculate forces
        self.calculate_total_forces()
        
        # Update velocities and positions using Euler method
        for obj in self.objects:
            # a = F/m
            acceleration = obj.force / obj.mass
            
            # Update velocity: v = v + a*dt
            obj.velocity += acceleration * dt
            
            # Update position: x = x + v*dt
            obj.position += obj.velocity * dt
            
        self.time += dt
        
    def get_kinetic_energy(self) -> float:
        """Calculate total kinetic energy of the system."""
        total_ke = 0
        for obj in self.objects:
            v_squared = np.dot(obj.velocity, obj.velocity)
            total_ke += 0.5 * obj.mass * v_squared
        return total_ke
        
    def get_potential_energy(self) -> float:
        """Calculate total gravitational potential energy of the system."""
        total_pe = 0
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects[i+1:], i+1):
                r = np.linalg.norm(obj2.position - obj1.position)
                if r > 0:  # Avoid division by zero
                    total_pe -= self.G * obj1.mass * obj2.mass / r
        return total_pe
        
    def get_total_energy(self) -> float:
        """Calculate total energy (kinetic + potential) of the system."""
        return self.get_kinetic_energy() + self.get_potential_energy()
        
    def get_center_of_mass(self) -> np.ndarray:
        """Calculate the center of mass of the system."""
        total_mass = sum(obj.mass for obj in self.objects)
        if total_mass == 0:
            return np.zeros(2)
            
        weighted_positions = sum(obj.mass * obj.position for obj in self.objects)
        return weighted_positions / total_mass
        
    def enable_visualization(self, trail_length: int = 100, figsize: tuple = (10, 10)):
        """
        Enable real-time visualization of the simulation.
        
        Args:
            trail_length: Number of previous positions to show as trails
            figsize: Figure size (width, height)
        """
        self.visualization_enabled = True
        self.trail_length = trail_length
        
        # Calculate fixed limits based on initial object positions and estimated orbits
        if self.objects:
            positions = np.array([obj.position for obj in self.objects])
            velocities = np.array([obj.velocity for obj in self.objects])
            
            # Estimate maximum distance by considering initial positions and velocities
            # Use a factor to account for orbital motion
            max_distance = np.max(np.linalg.norm(positions, axis=1))
            max_velocity = np.max(np.linalg.norm(velocities, axis=1))
            
            # Rough estimate of orbital radius based on velocity and distance
            # Add a safety margin of 1.5x
            estimated_range = max(max_distance, max_distance * 1.2) * 1.5
            
            self.fixed_limits = {
                'xlim': (-estimated_range, estimated_range),
                'ylim': (-estimated_range, estimated_range)
            }
        else:
            # Default limits if no objects exist yet
            default_range = 3e11  # ~2 AU
            self.fixed_limits = {
                'xlim': (-default_range, default_range),
                'ylim': (-default_range, default_range)
            }
        
        # Initialize matplotlib figure
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Physics Simulation')
        
        # Set the fixed limits
        self.ax.set_xlim(self.fixed_limits['xlim'])
        self.ax.set_ylim(self.fixed_limits['ylim'])
        
        # Initialize trails for each object
        for obj in self.objects:
            self.trails[obj.name] = []
            
    def disable_visualization(self):
        """Disable visualization and close the plot window."""
        self.visualization_enabled = False
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
        self.trails.clear()
        
    def update_visualization(self):
        """Update the visualization with current object positions."""
        if not self.visualization_enabled or self.ax is None:
            return
            
        # Clear the plot
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f'Physics Simulation - Time: {self.time:.2e} s')
        
        # Update trails and plot objects
        for obj in self.objects:
            # Add current position to trail
            if obj.name not in self.trails:
                self.trails[obj.name] = []
                
            self.trails[obj.name].append(obj.position.copy())
            
            # Keep trail length limited
            if len(self.trails[obj.name]) > self.trail_length:
                self.trails[obj.name].pop(0)
                
            # Plot trail
            if len(self.trails[obj.name]) > 1:
                trail_positions = np.array(self.trails[obj.name])
                self.ax.plot(trail_positions[:, 0], trail_positions[:, 1], 
                           alpha=0.5, linewidth=1, linestyle='-')
                           
            # Plot object (size based on mass)
            size = max(10, min(200, np.log10(obj.mass) * 5))
            self.ax.scatter(obj.position[0], obj.position[1], s=size, 
                          label=obj.name if obj.name else f"Object {id(obj)}")
                          
        # Use fixed axis limits (set during enable_visualization)
        if self.fixed_limits:
            self.ax.set_xlim(self.fixed_limits['xlim'])
            self.ax.set_ylim(self.fixed_limits['ylim'])
                           
        self.ax.legend()
        plt.draw()
        plt.pause(0.01)  # Small pause to allow plot to update
        
    def run_simulation(self, dt: float, steps: int, update_interval: int = 1):
        """
        Run the simulation for a specified number of steps with optional visualization.
        
        Args:
            dt: Time step (s)
            steps: Number of simulation steps
            update_interval: Update visualization every N steps (ignored if visualization disabled)
        """
        for step in range(steps):
            self.update_physics(dt)
            
            if self.visualization_enabled and step % update_interval == 0:
                self.update_visualization()
                
    def reset(self):
        """Reset the simulation (clear all objects and reset time)."""
        self.objects.clear()
        self.time = 0.0
        self.trails.clear()
        self.fixed_limits = None


if __name__ == "__main__":
    # Example usage with visualization
    sim = PhysicsSimulator()
    
    # Add some objects (using scaled units for better visualization)
    sun = sim.add_object(mass=1.989e30, position=[0, 0], velocity=[0, 0], name="Sun")
    earth = sim.add_object(mass=5.972e24, position=[1.496e11, 0], velocity=[0, 29780], name="Earth")
    mars = sim.add_object(mass=6.39e23, position=[2.279e11, 0], velocity=[0, 24070], name="Mars")
    
    print("Initial state:")
    print(f"Sun: {sun}")
    print(f"Earth: {earth}")
    print(f"Mars: {mars}")
    print(f"Force on Earth: {sim.get_force_on_object(earth)}")
    
    # Enable visualization
    sim.enable_visualization(trail_length=50)
    
    # Run simulation with visualization
    dt = 86400 * 5  # 5 days per step
    print(f"\nRunning simulation with visualization...")
    print("Close the plot window to end the simulation.")
    
    try:
        sim.run_simulation(dt=dt, steps=365, update_interval=1)
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        sim.disable_visualization()
        
    print("Simulation complete!")