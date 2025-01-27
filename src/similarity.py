import numpy as np
from src.segmentation import Region

def compute_color_similarity(region1, region2):
    """Compute color similarity between two regions using histogram intersection
    
    Args:
        region1 (Region): First region
        region2 (Region): Second region
        color_space: Color space to compute similaritysimilarity
        
    Returns:
        float: Color similarity score between 0 and 1
        
    Note:
        Uses color histograms in all channels, normalized using L1 norm
    """
    hist1 = region1.color_hist
    hist2 = region2.color_hist
    
    return np.sum(np.minimum(hist1, hist2))

def compute_texture_similarity(region1, region2):
    """Compute texture similarity using SIFT-like measurements
    
    Args:
        region1 (Region): First region
        region2 (Region): Second region
        
    Returns:
        float: Texture similarity score between 0 and 1
        
    Note:
        Uses Gaussian derivatives in 8 orientations for each color channel
    """
    hist1 = region1.texture_hist
    hist2 = region2.texture_hist

    return np.sum(np.minimum(hist1, hist2))

def compute_size_similarity(region1, region2, image_size):
    """Compute size similarity to encourage merging small regions
    
    Args:
        region1 (Region): First region
        region2 (Region): Second region
        image_size (int): Total image size in pixels
        
    Returns:
        float: Size similarity score between 0 and 1
    """
    return 1.0 - (region1.size + region2.size) / float(image_size)

def compute_fill_similarity(region1, region2, image_size):
    """Compute fill similarity to encourage regions that fit well
    
    Args:
        region1 (Region): First region
        region2 (Region): Second region
        image_size (int): Total image size in pixels
        
    Returns:
        float: Fill similarity score between 0 and 1
        
    Note:
        Uses bounding box size vs region size
    """
    rmin1, rmax1, cmin1, cmax1 = region1.bbox
    rmin2, rmax2, cmin2, cmax2 = region2.bbox

    rmin = min(rmin1, rmin2)
    rmax = max(rmax1, rmax2)
    cmin = min(cmin1, cmin2)
    cmax = max(cmax1, cmax2)

    merged_bbox_size = (rmax-rmin)*(cmax-cmin)
    region_size = region1.size + region2.size

    return 1.0 - (merged_bbox_size-region_size) / float(image_size)