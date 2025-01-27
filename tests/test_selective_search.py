import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.segmentation import *
from src.selective_search import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class BoxVisualizer:
   def __init__(self, image, boxes):
       self.image = image
       self.boxes = boxes
       self.idx = 0
       
       # Create figure for boxes
       self.fig_boxes = plt.figure(figsize=(8,8))
       self.ax_boxes = self.fig_boxes.add_subplot(111)
       self.btn_prev = Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Previous')
       self.btn_next = Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Next')
       
       # Connect buttons
       self.btn_prev.on_clicked(self.prev_box)
       self.btn_next.on_clicked(self.next_box)
       
       self.update_plot()
       
   def update_plot(self):
       self.ax_boxes.clear()
       self.ax_boxes.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
       
       # Draw current box
       box = self.boxes[self.idx]
       x1,y1,x2,y2 = box
       rect = plt.Rectangle((x1,y1), x2-x1, y2-y1, 
                          fill=False, color='red', linewidth=2)
       self.ax_boxes.add_patch(rect)
       
       self.ax_boxes.set_title(f'Box {self.idx+1}/{len(self.boxes)}')
       self.ax_boxes.axis('off')
       self.fig_boxes.canvas.draw_idle()
       
   def prev_box(self, event):
       self.idx = (self.idx - 1) % len(self.boxes)
       self.update_plot()
       
   def next_box(self, event):
       self.idx = (self.idx + 1) % len(self.boxes)
       self.update_plot()

def visualize_results(image, boxes, labels=None):
   """
   Visualize selective search results with segments and interactive boxes
   """
   # Create figure for original & segments
   fig = plt.figure(figsize=(12,5))
   
   # Original image
   plt.subplot(121)
   plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
   plt.title('Original Image')
   plt.axis('off')
   
   # Segmentation 
   if labels is not None:
       plt.subplot(122)
       n_segments = len(np.unique(labels))
       colors = np.random.randint(0, 255, size=(n_segments, 3), dtype=np.uint8)
       segm_image = colors[labels]
       plt.imshow(segm_image)
       plt.title(f'Initial Segments: {n_segments}')
       plt.axis('off')
   
   plt.tight_layout()
   
   # Create interactive box visualization
   box_vis = BoxVisualizer(image, boxes)
   
   plt.show()

if __name__ == "__main__":
   # Load image
   image = cv2.imread("./data/BSD500/images/train/8049.jpg")
   
   # Run selective search
   ss = SelectiveSearch()
   
   # Get initial segmentation
   labels = felzenszwalb_segmentation(image, k=100) 
   
   # Get proposals
   boxes, scores = ss.generate_proposals(image)
   
   # Visualize
   visualize_results(image, boxes, labels)