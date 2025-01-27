import numpy as np

def extract_boxes(regions):
   """Convert regions to boxes"""
   boxes = []
   for region in regions:
       rmin, rmax, cmin, cmax = region.bbox
       boxes.append([cmin, rmin, cmax, rmax])
   return np.array(boxes)

def compute_iou(box1, box2):
   """Compute intersection over union between boxes"""
   x11, y11, x12, y12 = box1
   x21, y21, x22, y22 = box2
   
   xi1 = max(x11, x21)
   yi1 = max(y11, y21)
   xi2 = min(x12, x22)
   yi2 = min(y12, y22)
   
   intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
   box1_area = (x12 - x11) * (y12 - y11)
   box2_area = (x22 - x21) * (y22 - y21)
   
   union = box1_area + box2_area - intersection
   return intersection / union

def non_max_suppression(boxes, scores, iou_threshold=0.5):
   """Filter out overlapping boxes"""
   idxs = np.argsort(scores)[::-1]
   keep = []
   
   while idxs.size > 0:
       keep.append(idxs[0])
       ious = np.array([compute_iou(boxes[idxs[0]], boxes[i]) for i in idxs[1:]])
       idxs = idxs[1:][ious <= iou_threshold]
       
   return keep