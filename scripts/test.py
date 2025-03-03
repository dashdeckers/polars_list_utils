import numpy as np
import matplotlib.pyplot as plt

def compute_point_to_line_distances(points, anchor1, anchor2):
    """
    Compute the perpendicular distance from each point to a line defined by two anchor points.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of points with shape (n, 2) where n is the number of points
    anchor1 : numpy.ndarray
        First anchor point with shape (2,)
    anchor2 : numpy.ndarray
        Second anchor point with shape (2,)
    
    Returns:
    --------
    distances : numpy.ndarray
        Array of distances with shape (n,)
    """
    # Convert inputs to numpy arrays if they aren't already
    points = np.asarray(points)
    anchor1 = np.asarray(anchor1)
    anchor2 = np.asarray(anchor2)
    
    # Step 1: Calculate the direction vector of the line
    line_vector = anchor2 - anchor1
    
    # Step 2: Normalize the line vector to get a unit vector
    line_unit_vector = line_vector / np.linalg.norm(line_vector)
    
    # Step 3: For each point, compute the vector from anchor1 to the point
    point_vectors = points - anchor1
    
    # Step 4: Project each point vector onto the line vector
    # The dot product gives the scalar projection (how far along the line)
    dot_products = np.dot(point_vectors, line_unit_vector)
    
    # Step 5: Calculate the projection points on the line
    projection_points = anchor1 + np.outer(dot_products, line_unit_vector)
    
    # Step 6: Calculate the vector from each point to its projection
    distance_vectors = points - projection_points
    
    # Step 7: Calculate the norm (length) of each distance vector
    distances = np.linalg.norm(distance_vectors, axis=1)
    
    return distances, projection_points


# Example usage with 1D Fourier Transform data
# Let's simulate some data for demonstration
def example_with_1d_data():
    # Generate x values (indices)
    x = np.arange(100)
    
    # Generate y values (simulated Fourier transform magnitude)
    # Create a continuous line with some noise
    y = 10 * np.sin(x * 0.1) + np.random.normal(0, 1, 100)
    
    # Combine x and y to form 2D points
    points = np.column_stack((x, y))
    
    # Define anchor points (selecting points at indices 20 and 80)
    anchor1 = points[20]
    anchor2 = points[80]
    
    # Compute distances
    distances, projections = compute_point_to_line_distances(points, anchor1, anchor2)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Original Data')
    plt.plot([anchor1[0], anchor2[0]], [anchor1[1], anchor2[1]], 'r-', label='Fitted Line')
    
    # Draw perpendicular lines from some points to the line
    for i in range(20, 81, 10):
        plt.plot([points[i][0], projections[i][0]], 
                 [points[i][1], projections[i][1]], 
                 'g--', alpha=0.5)
    
    plt.scatter([anchor1[0], anchor2[0]], [anchor1[1], anchor2[1]], color='red', s=100, label='Anchor Points')
    plt.scatter(x[20:81:10], y[20:81:10], color='green', s=50, label='Sample Points')
    
    plt.title('Distance from Points to Line')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Distances for points between anchors (indices 20-80): {distances[20:81]}")
    
    return distances

# Run the example
if __name__ == "__main__":
    distances = example_with_1d_data()