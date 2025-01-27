import cv2
from src.segmentation import felzenszwalb_segmentation
import numpy as np
import matplotlib.pyplot as plt

def visualize_segments(image, labels):
   """
   Visualize segmentation results
   
   Args:
       image: Original RGB image
       labels: Segmentation labels from felzenszwalb
   """
   # Generate random colors for segments
   n_segments = len(np.unique(labels))
   colors = np.random.randint(0, 255, size=(n_segments, 3), dtype=np.uint8)
   
   # Create segmentation visualization
   segm_image = colors[labels]
   
   # Create boundaries
   boundaries = np.zeros_like(image)
   for i in range(1, labels.shape[0]):
       for j in range(1, labels.shape[1]):
           if labels[i,j] != labels[i-1,j] or labels[i,j] != labels[i,j-1]:
               boundaries[i,j] = [255,255,255]
               
   # Plot results
   plt.figure(figsize=(15,5))
   
   plt.subplot(131)
   plt.imshow(image)
   plt.title('Original Image')
   plt.axis('off')
   
   plt.subplot(132) 
   plt.imshow(segm_image)
   plt.title(f'Segments: {n_segments}')
   plt.axis('off')
   
   plt.subplot(133)
   plt.imshow(boundaries)
   plt.title('Boundaries')
   plt.axis('off')
   
   plt.tight_layout()
   plt.show()
image = cv2.imread('./data/BSD500/images/train/8049.jpg') 
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
labels = felzenszwalb_segmentation(image, k=100, min_size=50)
print(labels)

visualize_segments(image, labels)