import cv2
import argparse
from yoloseg import YOLOSeg
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SamProcessor, SamModel, SamConfig
import cv2
import io
import onnxruntime
import time

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOSeg Inference")
    parser.add_argument("--type", type=str, default="yolo", help="which model you want [yolo , sam ,unet]")
    parser.add_argument("--model", type=str, default="models/Yolov8L.onnx", help="Path to the ONNX model file")
    parser.add_argument("--image", type=str, default="C:/Users/WB GAMING/Desktop/training/test.jpg", help="Path to the input image")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Confidence threshold for object detection")
    parser.add_argument("--iou_thres", type=float, default=0.3, help="IOU threshold for non-maximum suppression")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.type == "sam":

        model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        my_mito_model = SamModel(config=model_config)
        my_mito_model.load_state_dict(torch.load(args.model))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        my_mito_model.to(device)

        image_pil = Image.open(args.image)

        with io.BytesIO() as output:
            image_pil.save(output, format="TIFF")
            tiff_bytes = output.getvalue()

        large_test_images = np.array(Image.open(io.BytesIO(tiff_bytes)))

        if len(large_test_images.shape) == 2:
            large_test_images = np.stack((large_test_images,) * 3, axis=-1)

        image_patches = []
        for i in range(0, large_test_images.shape[0], 640):
            for j in range(0, large_test_images.shape[1], 640):
                patch = large_test_images[i:i+640, j:j+640]
                image_patches.append(patch)

        grid_size = 10
        x = np.linspace(0, 639, grid_size)
        y = np.linspace(0, 639, grid_size)
        xv, yv = np.meshgrid(x, y)

        input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv, yv)]
        input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)

        for patch in image_patches:
            single_patch = Image.fromarray(patch)
            inputs = processor(single_patch, input_points=input_points, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = my_mito_model(**inputs, multimask_output=False)

            start_time = time.time()
            single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1)).cpu().numpy().squeeze()
            end_time = time.time()
            elapsed_time = end_time - start_time

            single_patch_prediction = (single_patch_prob > 0.9).astype(np.uint8)


            single_patch_prediction_resized = cv2.resize(single_patch_prediction, (640, 640))


            overlay = np.zeros_like(patch)
            overlay[single_patch_prediction_resized == 1] = [0, 0, 255] 

            print("Elapsed time:", elapsed_time, "seconds")
            plt.figure(figsize=(10, 5))
            plt.imshow(patch)
            plt.imshow(overlay, alpha=0.6)
            plt.title("Segmentation Overlay")
            plt.axis('off')
            plt.show()


    elif args.type == "yolo":
        yoloseg = YOLOSeg(args.model, conf_thres=args.conf_thres, iou_thres=args.iou_thres)

        img = cv2.imread(args.image)

        start_time = time.time()
        boxes, scores, class_ids, masks = yoloseg(img)
        end_time = time.time()
        elapsed_time = end_time - start_time

        combined_img = yoloseg.draw_masks(img)

        print(f"Number of wheat detected: {len(boxes)}")
        print("Elapsed time:", elapsed_time, "seconds")

        cv2.imshow("Wheat Detected", combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.type == "unet":

        model_path = args.model
        image_path = args.image

        img = cv2.imread(image_path)
        H, W, _ = img.shape

        img_resized = cv2.resize(img, (128, 128))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, ...]

        expected_input_shape = (1, 128, 128, 1)
        img_input = np.resize(img_input, expected_input_shape)

        start_time = time.time()
        ort_session = onnxruntime.InferenceSession(model_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        outputs = ort_session.run(None, {'input_2': img_input})
        masks = outputs[0][0] * 255

        masks_resized = cv2.resize(masks, (W, H), interpolation=cv2.INTER_LINEAR)
        _, binary_mask = cv2.threshold(masks_resized, 128, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        result_image = img.copy()
        for idx, contour in enumerate(contours):
            if cv2.contourArea(contour) > 0.5:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                polygon = cv2.approxPolyDP(contour, epsilon, True)
                x, y, w, h = cv2.boundingRect(contour)
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                cv2.fillPoly(result_image, [polygon], color)
        print(f"Number of wheat detected: {len(contour)}")
        print("Elapsed time:", elapsed_time, "seconds")
        cv2.imshow('Detection wheat', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
