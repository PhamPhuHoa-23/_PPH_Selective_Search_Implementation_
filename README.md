# Selective Search for Object Recognition

This is a Python implementation of the "Selective Search for Object Recognition" paper by J.R.R. Uijlings et al. 

## Overview

Selective Search is an object proposal algorithm that combines the strength of both exhaustive search and segmentation. It uses image structure to generate object location proposals quickly and effectively.

## Key Components

1. **Initial Segmentation**
- Uses Felzenszwalb's graph-based segmentation
- Creates initial regions at different scales

2. **Hierarchical Grouping**
- Iteratively merges similar regions
- Uses multiple complementary grouping criteria:
  - Color similarity (using histogram intersection)
  - Texture similarity (using SIFT-like features)
  - Size similarity (encourages merging small regions)
  - Fill similarity (measures how well regions fit into each other)

3. **Diversification Strategies** 
- Multiple color spaces (RGB, HSV, Lab, rgI)
- Complementary similarity measures
- Various starting scales

## Project Structure

```
selective_search/
├── data/
│   └── BSD500/images/train/  # Test images (BSD500 Dataset)
├── src/
│   ├── selective_search.py   # Main algorithm implementation
│   ├── color_spaces.py       # Color space conversions
│   ├── similarity.py         # Similarity measures
│   ├── segmentation.py       # Initial segmentation 
│   └── boxes.py              # Bounding box utilities
├── tests/
│   └── test_selective_search.py
└── requirements.txt
```

## Requirements

- Python 3.6+
- NumPy
- OpenCV
- SciPy
- Matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from selective_search import SelectiveSearch

# Load image
image = cv2.imread("image.jpg")

# Initialize selective search
ss = SelectiveSearch()

# Generate object proposals
boxes, scores = ss.generate_proposals(image)
```

## Parameters

- `base_k`: Base scale parameter for initial segmentation (default: 150)
- `min_size`: Minimum component size (default: 100)
- `sigma`: Gaussian filter parameter (default: 0.8)

## Testing
```python
python -m tests.test_selective_search
```

## Visualization

```python
from tests.test_selective_search import visualize_results

# Visualize results
visualize_results(image, boxes, labels)
```

## Reference

[1] J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders.
"Selective Search for Object Recognition"
International Journal of Computer Vision, 2013.

## License

MIT License