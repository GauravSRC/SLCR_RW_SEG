# River Water Segmentation using PyTorch
This repository contains a deep learning model for river water segmentation from aerial and satellite imagery. The model uses a MobileNetV3-based architecture to identify and segment water bodies in images and videos.
## Project Overview
This project implements a semantic segmentation model that can detect river water in images and video. The model was trained on the RiWa (River Water) dataset and can be used for:
- River monitoring
- Water body detection
- Environmental analysis
- Water resource management
## Model Architecture
The model architecture consists of:
- **Backbone**: MobileNetV3 Large (pretrained on ImageNet)
- **Decoder**: Custom decoder with transposed convolutions for upsampling
- **Output**: Single-channel segmentation mask with sigmoid activation
The model is lightweight and optimized for fast inference while maintaining accuracy.
## Dataset
DATASET LINK  (kaggle)=>** https://www.kaggle.com/datasets/franzwagner/river-water-segmentation-dataset**
The model was trained on the River Water Segmentation Dataset (RiWa v2), which contains:
- RGB aerial/satellite images of rivers
- Binary segmentation masks
- Training and validation splits
## Training Results
**The model was trained for 10 epochs(executed it roughly (will improve the code and upload version 2 later)).
**
The model achieved the lowest validation loss of 0.1921 at epoch 4.
## Features
- **Image Segmentation**: Process individual images to detect river water
- **Video Processing**: Segment rivers in video footage with post-processing techniques
- **Post-processing**: Includes erosion and area-based filtering to reduce noise
## Requirements
- Python 3.6+
- PyTorch 1.7+
- OpenCV
- NumPy
- Albumentations
- Matplotlib
- tqdm
- scikit-learn
- CUDA-capable GPU (recommended)
## Usage
### Image Segmentation
```python
# Load the trained model
model = SegmentationModel().to(device)
model.load_state_dict(torch.load('best_model.pth'))

# Process a single image
process_image(model, "path/to/your/image.jpg")
```
### Video Processing
```python
# Load the trained model
model = SegmentationModel().to(device)
model.load_state_dict(torch.load('best_model.pth'))

# Process a video
input_video_path = 'path/to/input/video.mp4'
output_video_path = 'path/to/output/video.mp4'

# Parameters for post-processing
min_area = 500  # Minimum area to keep regions
kernel_size = 5  # Erosion kernel size

# Run video processing pipeline
# (See code for complete implementation)
```
## Model Performance
The model implements several techniques to improve performance:
- Data augmentation (horizontal flips, rotations, brightness/contrast adjustments)
- Post-processing with erosion to remove noise
- Area-based filtering to remove small false positives
- Adjustable threshold for binary segmentation
## Future Work
- Implementation of more advanced architectures (U-Net, DeepLabV3+)
- Exploration of attention mechanisms for improved boundary detection
- Training on larger and more diverse datasets
- Extension to multi-class segmentation (differentiating between river, lake, and ocean water)
- Model quantization for deployment on edge devices
## License
[MIT License](LICENSE)
## Acknowledgements
- The River Water Segmentation Dataset creators
- PyTorch team for their excellent deep learning framework
- Albumentations library for efficient data augmentation
