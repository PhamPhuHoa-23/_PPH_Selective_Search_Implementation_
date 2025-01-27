import numpy as np
import cv2
from dataclasses import dataclass
from scipy.sparse import csc_matrix
from sklearn.feature_extraction import image

@dataclass
class Edge():
    weight: float
    a: int
    b: int

class UnionFind():
    def __init__(self, num_elements):
        self.num_elements = num_elements
        self.elts = np.arange(num_elements)
        self.sizes = np.ones(num_elements)

    def find(self, x):
        y = x
        while self.elts[y] != y:
            y = self.elts[y]
        self.elts[y] = y
        return y
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            if self.sizes[root_x] < self.sizes[root_y]:
                root_x, root_y = root_y, root_x
            self.elts[root_y] = root_x
            self.sizes[root_x] += self.sizes[root_y]

def _build_graph(img, height, width):
    # Tạo edge giữa các pixel lân cận
    edges = []

    # Tính diffirence giữa pixel lân cận
    for y in range(height):
        for x in range(width):
            vertex_id = y*width + x

            if x < width - 1: # Right neighbor
                w = np.sum((img[y,x] - img[y,x+1])**2)
                edges.append(Edge(w, vertex_id, vertex_id+1))
            
            if y < height - 1: # Bottom neighbor
                w = np.sum((img[y,x] - img[y+1,x])**2)
                edges.append(Edge(w, vertex_id, vertex_id+width))

            if x < width - 1 and y < height - 1: # Bottom-right neighbor
                w = np.sum((img[y,x] - img[y+1,x+1])**2)
                edges.append(Edge(w, vertex_id, vertex_id+width+1))

            if x > 0 and y < height - 1: # Bottom-left neighbor
                w = np.sum((img[y,x] - img[y+1,x-1]))
                edges.append(Edge(w, vertex_id, vertex_id+width-1))
    
    return sorted(edges, key=lambda x: x.weight)

def felzenszwalb_segmentation(image, k=100, min_size=100):
    """Perform initial segmentation using Felzenszwalb algorithm
    
    Args:
        image (ndarray): Input image with shape (H, W, 3)
        k (int): Scale parameter that controls segment size
        min_size (int): Minimum component size
        
    Returns:
        ndarray: Segmentation mask where each unique value represents a segment
        List[Region]: List of Region objects containing segment properties
        
    Note:
        This creates initial regions that will be merged in hierarchical grouping
    """
    height, width = image.shape[:2]
    num_vertices = height * width

    edges = _build_graph(image, height, width)

    forest = UnionFind(num_elements=num_vertices)
    threshold = np.zeros(num_vertices)

    for edge in edges:
        a = forest.find(edge.a)
        b = forest.find(edge.b)

        if a != b:
            weight = edge.weight
            ta = threshold[a]
            tb = threshold[b]
            if (weight <= ta and weight <= tb):
                forest.union(a, b)
                parent = forest.find(a)
                threshold[parent] = weight + k/forest.sizes[parent]

    for edge in edges:
        a = forest.find(edge.a)
        b = forest.find(edge.b)

        if a != b and (forest.sizes[a] < min_size or forest.sizes[b] < min_size):
            forest.union(a, b)

    labels = np.zeros(num_vertices, dtype=np.int32)
    for i in range(num_vertices):
        labels[i] = forest.find(i)

    labels = labels.reshape(height, width)
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        labels[labels == label] = i

    return labels


class Region:
    """Class to store region properties"""
    def __init__(self, mask, image):
        """Initialize region from mask
        
        Args:
            mask (ndarray): Binary mask of region
            image (ndarray): Original image
        """
        self.mask = mask
        self.bbox = self._compute_bbox()
        self.size = np.sum(mask)

        self.color_hist = self._compute_color_hist(image)

        self.texture_hist = self._compute_texture_hist(image)

    def _compute_bbox(self):
        """Calculate bounding box from mask"""
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return (rmin, rmax, cmin, cmax)
    
    def _compute_color_hist(self, image, n_bins=25):
        """Compute color histogram for each channel"""
        if image is None: return
        masked_img = image[self.mask]

        hist = np.array([
            np.histogram(masked_img[:, i], bins=n_bins, range=(0,255))[0]
            for i in range(3)
        ])

        hist = hist.astype(float) / hist.sum()
        return hist
    
    def _compute_texture_hist(self, image, n_orientations=8):
        """Compute texture histogram using Gaussian deriviatives"""
        if image is None: return None
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        texture_hist = []
        for sigma in [1.0, 2.0]:
            # Gaussian smoothing
            smoothed = cv2.GaussianBlur(gray, (0,0), sigma)
            
            # Compute derivatives
            dx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)

            # Magnitude and orientation
            magnitude = np.sqrt(dx**2 + dy**2)
            orientation = np.arctan2(dy, dx) * (180 / np.pi) % 180

            # Compute histogram
            hist = np.zeros(n_orientations)
            for i in range(n_orientations):
                angle = i * 180 / n_orientations
                angle_range = 180 / n_orientations

                # Find pixels within angle range
                idx = np.logical_and(
                    orientation >= angle - angle_range/2,
                    orientation < angle + angle_range/2
                )
                
                # Weight by magnitude and mask
                hist[i] = np.sum(magnitude[idx & self.mask])

            texture_hist.extend(hist)

        # Normalize
        texture_hist = np.array(texture_hist)
        if np.sum(texture_hist) > 0:
            texture_hist = texture_hist / np.sum(texture_hist)
            
        return texture_hist
    
    def merge(self, other):
        """Merge with another region"""

        """
        Args:
            other(Region): Region to merge with

        Returns:
            Region: New merged region
        """

        """Merge with another region using weighted histograms"""
        new_mask = self.mask | other.mask
        
        # Update color histogram using weights
        w1 = float(self.size) / (self.size + other.size)
        w2 = float(other.size) / (self.size + other.size)
        new_color_hist = w1 * self.color_hist + w2 * other.color_hist
        
        # Update texture histogram
        new_texture_hist = w1 * self.texture_hist + w2 * other.texture_hist
        
        # Create new region but avoid recomputing histograms
        merged = Region(new_mask, None)
        if self.color_hist is not None and other.color_hist is not None:
            merged.color_hist = w1 * self.color_hist + w2 * other.color_hist
            
        if self.texture_hist is not None and other.texture_hist is not None:
            merged.texture_hist = w1 * self.texture_hist + w2 * other.texture_hist
            
        return merged