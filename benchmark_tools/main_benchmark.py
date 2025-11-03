import numpy as np
from ultralytics import YOLO
import imagesize
import argparse
from yolo_attnopt.attention_optimization import setup_optimized_yolo_environ


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for YOLO object detection.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='YOLO object detection script')
    parser.add_argument('--optimize-level', type=int, default=0,
                      help='Optimization level (default: 0)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                      default='cuda', help='Device to run on (default: cuda)')
    parser.add_argument('--yolo-modelname', type=str, default='yolo11m.pt',
                      help='YOLO model name (default: yolo11l.pt)')
    parser.add_argument('--image-scale-down', type=int, default=32,
                      help='Factor to scale down image dimensions (default: 32)')
    parser.add_argument('--image-scale-up', type=int, default=32,
                      help='Factor to scale up image after scaling down (default: 32)')
    parser.add_argument('--confidence-threshold', type=float, default=0.13,
                      help='Minimum confidence threshold for detections (default: 0.13)')
    parser.add_argument('--save-images', action='store_true',
                      help='Save detected images (default: False)')
    parser.add_argument('--print-bbox', action='store_true',
                      help='Print bounding box coordinates (default: False)')
    parser.add_argument('--image-path', type=str, required=False,
                      help='Path to the input image (optional)')
    parser.add_argument('--image-size', type=int, default=640,
                      help='Size N for random NxNx3 image when no image path provided (default: 640)')
    return parser.parse_args()

def main(yolo_modelname: str, image_scale_down: int, image_scale_up: int, 
         confidence_threshold: float, save_images: bool, print_bbox: bool, 
         image_path: str, optimize_level: int, device: str, image_size: int) -> None:
    """
    Main function to perform object detection using YOLO.
    
    Args:
        yolo_modelname: Name of the YOLO model file
        image_scale_down: Factor to scale down image dimensions
        image_scale_up: Factor to scale up image after scaling down
        confidence_threshold: Minimum confidence threshold for detections
        save_images: Whether to save detected images
        print_bbox: Whether to print bounding box coordinates
        image_path: Path to the input image
        optimize_level: Optimization level for YOLO processing
        device: Device to run on ('cuda' or 'cpu')
        image_size: Size N for random NxNx3 image when no image path provided
    """
    # Print input arguments
    print("\n[INFO] Input Arguments:")
    print(f"  yolo_modelname: {yolo_modelname}")
    print(f"  image_scale_down: {image_scale_down}")
    print(f"  image_scale_up: {image_scale_up}")
    print(f"  confidence_threshold: {confidence_threshold}")
    print(f"  save_images: {save_images}")
    print(f"  print_bbox: {print_bbox}")
    print(f"  image_path: {image_path}")
    print(f"  optimize_level: {optimize_level}")
    print(f"  device: {device}")
    print(f"  image_size: {image_size}\n")

    # Apply the optimization patch to ultralytics
    print("[INFO] Applying optimization patch to ultralytics...")
    
    # Patch it!
    setup_optimized_yolo_environ(optimize_level=optimize_level, debug_mode=True)

    # Generate random image if no image path provided, otherwise get dimensions from file
    if image_path:
        W, H = imagesize.get(image_path)
        input_img = image_path
    else:
        W = H = image_size
        # Generate random RGB image with values between 0 and 255
        input_img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        
    input_img_size = [W, H]
        
    # Load and initialize YOLO model
    model = YOLO(yolo_modelname)
    model = model.to(device)

    # Calculate dimensions for YOLO processing
    # First scale down by dividing by image_scale_down, then scale up by multiplying by image_scale_up
    yolo_imgW = image_scale_up * int(np.floor(W / image_scale_down))
    yolo_imgH = image_scale_up * int(np.floor(H / image_scale_down))

    imgsz=[yolo_imgW, yolo_imgH]
    print('[INFO] Input image size:', input_img_size)
    print('[INFO] Running YOLO with imgsz argument:', imgsz)

    # Perform object detection
    results = model(input_img, imgsz=imgsz, conf=confidence_threshold)
    
if __name__ == '__main__':
    args = parse_args()
    main(args.yolo_modelname, args.image_scale_down, args.image_scale_up, 
         args.confidence_threshold, args.save_images, args.print_bbox, 
         args.image_path, args.optimize_level, args.device, args.image_size)
        