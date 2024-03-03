# Wheat Segmentation with YOLOSeg, SAM, and UNet

This Python script provides functionality to segment wheat from images using various segmentation models like YOLO, SAM, and UNet. It supports detecting and segmenting wheat instances from input images, providing overlays to visualize the segmentation results.

## Requirements

- Python 3
- OpenCV (`cv2`)
- YOLOSeg (`yoloseg`)
- PyTorch (`torch`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- Pillow (`PIL`)
- Transformers (`transformers`)
- ONNXRuntime (`onnxruntime`)

Install the required packages using the provided `requirements.txt` file:

```bash
cd Instance-Segmentation
pip install -r requirements.txt
```

## You can run the script with various options:

```bash
python image_instance_segmentation.py --type <model_type> --model <path_to_model> --image <path_to_image> --conf_thres <confidence_threshold> --iou_thres <iou_threshold>
```

## YOLO

```bash
python image_instance_segmentation.py --type yolo --model models/Yolov8L.onnx --image test.jpg --conf_thres 0.5 --iou_thres 0.3
```

## Sam

```bash
python image_instance_segmentation.py --type sam --model models/SAM.pth --image test.jpg
```

## U-Net

```bash
python image_instance_segmentation.py --type unet --model models/UNet.onnx --image test.jpg
```

## models

models drive link : https://drive.google.com/drive/folders/1L2xwdCekAJ5dP3_I2RPk1kcqfxp5pwua?usp=sharing