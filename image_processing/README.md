# Image Processing

This folder contains image processing algorithms and computer vision implementations.

## 📁 Contents

- **Image_processing.py** - Comprehensive image processing utilities and filters

## 🖼️ Image Processing Module

### Features
- Image filtering (Gaussian, Median, Bilateral)
- Edge detection (Canny, Sobel, Prewitt)
- Color space conversions (RGB, HSV, Grayscale)
- Image enhancement and restoration
- Morphological operations
- Histogram processing
- Contour detection
- Feature extraction

### Basic Usage
```bash
python Image_processing.py
```

### Code Examples

#### Edge Detection
```python
from Image_processing import edge_detect
edges = edge_detect(image, method='canny')
```

#### Image Filtering
```python
from Image_processing import apply_filter
filtered = apply_filter(image, filter_type='gaussian')
```

#### Color Conversion
```python
from Image_processing import convert_color
gray = convert_color(image, 'bgr2gray')
```

## 🎯 Common Tasks

### Blur Image
- Gaussian Blur (smoothing)
- Median Blur (noise removal)
- Motion Blur (dynamic effect)

### Detect Features
- Edges (Canny, Sobel)
- Corners (Harris)
- Contours
- Keypoints

### Enhance Image
- Contrast adjustment
- Histogram equalization
- Denoising
- Sharpening

## 📊 Supported Formats
- JPEG, PNG, BMP
- TIFF, GIF
- RAW image formats

## 🔧 Dependencies
- opencv-python (cv2)
- numpy
- matplotlib (visualization)
- PIL/Pillow (additional formats)

## 💡 Use Cases
- Medical image analysis
- Surveillance systems
- Object detection preprocessing
- Autonomous vehicle vision
- Document scanning
- Photo editing
- Quality inspection

## 📈 Performance Tips
- Use appropriate image resolution
- Preprocess images for faster computation
- Use GPU acceleration when available (CUDA)
- Batch process images for efficiency

## 🔗 References
- OpenCV Documentation: https://docs.opencv.org/
- Computer Vision Basics: https://en.wikipedia.org/wiki/Computer_vision
