import numpy as np
import cv2

def convert_to_hsv(image):
    """Convert RGB image to HSV color space
    
    Args:
        image (ndarray): Input RGB image with shape (H, W, 3)
        
    Returns:
        ndarray: HSV image with shape (H, W, 3)
        
    Note:
        HSV is useful for capturing color properties while being less sensitive to lighting changes
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    

def convert_to_lab(image):
    """Convert RGB image to Lab color space
    
    Args:
        image (ndarray): Input RGB image with shape (H, W, 3)
        
    Returns:
        ndarray: Lab image with shape (H, W, 3)
        
    Note:
        Lab separates luminance from color, making it robust to lighting changes
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    

def convert_to_rgI(image): 
    """Convert RGB image to normalized rgb + intensity color space
    
    Args:
        image (ndarray): Input RGB image with shape (H, W, 3)
        
    Returns:
        ndarray: Normalized rgb + intensity image with shape (H, W, 3)
        
    Note:
        Normalized rgb removes lighting intensity information while preserving color
    """
    # Calculate intensity
    rgb = image.astype("float32")
    intensity = np.mean(rgb, axis=2, keepdims=True)

    sum_rgb = np.sum(rgb, axis=2, keepdims=True)
    sum_rgb[sum_rgb == 0] = 1
    normalized_rgb = rgb / sum_rgb

    normalized_rgb[:,:,2] = intensity.squeeze()

    return normalized_rgb
    
def convert_to_rgb(image):
    """Convert image back to RGB color space
    
    Args:
        image (ndarray): Input image from another color space
        color_space (str): Source color space ('hsv', 'lab', 'rgI')
        
    Returns:
        ndarray: RGB image with shape (H, W, 3)
    """
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def convert_to_opponent(image):
    """Convert RGB to opponent color space
    
    Args:
        image (ndarray): Input RGB image
        
    Returns:
        ndarray: Opponent color space image
    """
    rgb = image.astype('float32')
    
    # O1 = (R-G)/sqrt(2)
    o1 = (rgb[:,:,0] - rgb[:,:,1])/np.sqrt(2)
    
    # O2 = (R+G-2B)/sqrt(6) 
    o2 = (rgb[:,:,0] + rgb[:,:,1] - 2*rgb[:,:,2])/np.sqrt(6)
    
    # O3 = (R+G+B)/sqrt(3)
    o3 = (rgb[:,:,0] + rgb[:,:,1] + rgb[:,:,2])/np.sqrt(3)
    
    opponent = np.stack([o1, o2, o3], axis=2)
    return opponent

def convert_to_normalized_rgb(image):
    """Convert to normalized RGB
    
    Args:
        image (ndarray): Input RGB image
        
    Returns:
        ndarray: Normalized RGB image
    """
    rgb = image.astype('float32')
    sum_rgb = np.sum(rgb, axis=2, keepdims=True)
    sum_rgb[sum_rgb == 0] = 1
    return rgb / sum_rgb