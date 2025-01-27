from scipy.ndimage import binary_dilation
from src.color_spaces import *
from src.similarity import *
from src.segmentation import *
from src.boxes import non_max_suppression
import numpy as np

class SelectiveSearch:
    def __init__(self, base_k=150, inc_k=150):
        self.base_k = base_k
        self.inc_k = inc_k
        self.color_spaces = {
            'rgb': convert_to_normalized_rgb,
            'hsv': convert_to_hsv, 
            'lab': convert_to_lab,
            'rgI': convert_to_rgI,
            'C': convert_to_opponent
        }
        
        # Similarity flags
        self.similarity_configs = [
            {'color': True, 'texture': True, 'size': True, 'fill': True},  # All
            {'color': True, 'texture': False, 'size': True, 'fill': True}, # No texture
            {'color': False, 'texture': True, 'size': True, 'fill': True}, # No color
            {'color': True, 'texture': True, 'size': False, 'fill': False} # Only color+texture
        ]

    def _calc_initial_similarities(self, regions, image_size):
        """Compute similarities only between adjacent regions"""
        similarities = {}
        for i, r1 in regions.items():
            for j, r2 in regions.items():
                if i >= j:
                    continue
                if self._are_adjacent(r1, r2):
                    sim = self._calc_similarity(r1, r2, image_size)
                    similarities[(i,j)] = sim
        return similarities


    def _are_adjacent(self, region1, region2):
        """
        Checking if region1 is next to region2

        Args:
            region1, region2: Region objects
        Returns:
            bool: True if region1 is next to region2
        """
        # Dialate first region's mask by 1 pixel
        dilated = binary_dilation(region1.mask)

        return np.any(dilated & region2.mask)

    def _calc_similarity(self, region1, region2, image_size):
        """
        Compute sum of 4 similarity

        Args:
            region1, region2: Region objects
            image_size: Size of input image

        Returns:
            floar: Sum of 4 similarity
        """
        similarity = 0.0
        if self.use_color:
            color_similarity = compute_color_similarity(region1, region2)
            similarity += color_similarity
        
        if self.use_texture:
            texture_similarity = compute_texture_similarity(region1, region2)
            similarity += texture_similarity
        
        if self.use_size:
            size_similarity = compute_size_similarity(region1, region2, image_size)
            similarity += size_similarity
        
        if self.use_fill:
            fill_similarity = compute_fill_similarity(region1, region2, image_size)
            similarity += fill_similarity

        return similarity

    def generate_proposals(self, image):
        """Generate proposals using multiple strategies"""
        boxes_all = []
        scores_all = []
        
        # For each color space
        for color_name, color_conv in self.color_spaces.items():
            # Convert image
            img_color = color_conv(image)
            
            # Multiple k values for initial segmentation 
            for k in [self.base_k, self.base_k + self.inc_k]:
                # Initial segmentation
                labels = felzenszwalb_segmentation(img_color, k)
                regions = {}
                for i, label in enumerate(np.unique(labels)):
                    regions[i] = Region(labels==label, image)

                # For each similarity config
                for sim_config in self.similarity_configs:
                    self.use_color = sim_config['color']
                    self.use_texture = sim_config['texture'] 
                    self.use_size = sim_config['size']
                    self.use_fill = sim_config['fill']

                    # Calculate initial similarities
                    self.image_size = image.shape[0] * image.shape[1]
                    self.max_size = 0.8 * self.image_size
                    similarities = self._calc_initial_similarities(regions, self.image_size)

                    # Hierarchical grouping
                    merged_regions = self._hierarchical_grouping(regions, similarities)

                    # Extract boxes
                    for region in merged_regions.values():
                        rmin, rmax, cmin, cmax = region.bbox
                        boxes_all.append([cmin, rmin, cmax, rmax])
                        scores_all.append(region.size)

            # break

        # Convert to numpy arrays
        boxes_all = np.array(boxes_all)
        scores_all = np.array(scores_all)

        # Non-maximum suppression
        idxs = non_max_suppression(boxes_all, scores_all, iou_threshold=0.5)
        
        return boxes_all[idxs], scores_all[idxs]

    def _hierarchical_grouping(self, regions, similarities):
        """Hierarchical grouping while keeping all scales"""
        all_regions = regions.copy()
        curr_idx = max(regions.keys()) + 1

        while similarities:
            print(len(regions), len(similarities))
            # Get most similar pair
            (i, j), highest_sim = max(similarities.items(), key=lambda x: x[1])
            
            # Try merging
            new_region = regions[i].merge(regions[j])
            if new_region.size > self.max_size:
                similarities.pop((i,j)) 
                break

            # Add new region but keep old ones
            regions[curr_idx] = new_region
            all_regions[curr_idx] = new_region

            regions.pop(i)
            regions.pop(j)

            # Update similarities
            to_remove = [(r1,r2) for (r1,r2) in similarities 
                        if r1 in (i,j) or r2 in (i,j)]
            for key in to_remove:
                similarities.pop(key)

            # Add new similarities
            for idx in regions:
                if idx != curr_idx and self._are_adjacent(regions[idx], new_region):
                    similarities[(idx, curr_idx)] = self._calc_similarity(
                        regions[idx], new_region, self.image_size)

            curr_idx += 1

        return all_regions