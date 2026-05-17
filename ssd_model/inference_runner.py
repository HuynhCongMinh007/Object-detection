import os
import json
import argparse
import sys
import torch
import numpy as np
from PIL import Image

from inference import load_model, run_inference, load_class_names_from_coco, draw_detections_opencv, NUM_CLASSES


def pick_sample_image(test_dir: str) -> str:
    files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        raise FileNotFoundError(f"No images found in {test_dir}")
    # pick the first image
    return os.path.join(test_dir, files[0])


def parse_args():
    p = argparse.ArgumentParser(description='Run SSD inference on a test image')
    p.add_argument('--image', '-i', help='Image filename (in test folder) or full path')
    p.add_argument('--open', '-o', action='store_true', help='Open annotated image after saving (Windows)')
    return p.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dir = os.path.dirname(__file__)
    test_dir = os.path.join(base_dir, 'animal_dataset', 'test')
    test_dir_image = os.path.join(base_dir, '', '')
    model_path = os.path.join(base_dir, 'checkpoints', 'best_ssd.pth')
    ann_path = os.path.join(test_dir, '_annotations.coco.json')

    print(f"Using device: {device}")

    # Resolve sample image
    sample_image = None
    if args.image:
        # If absolute or relative path provided and exists, use it
        if os.path.isabs(args.image) and os.path.exists(args.image):
            sample_image = args.image
        else:
            # try relative to test_dir
            candidate = os.path.join( test_dir_image, args.image)
            if os.path.exists(candidate):
                sample_image = candidate
            else:
                # try matching filename in test_dir (case-sensitive exact match)
                files = [f for f in os.listdir( test_dir_image) if f == args.image]
                if files:
                    sample_image = os.path.join( test_dir_image, files[0])
                else:
                    print(f"Error: image '{args.image}' not found in test folder: { test_dir_image}")
                    sys.exit(1)
    else:
        sample_image = pick_sample_image( test_dir_image)

    print(f"Selected sample image: {sample_image}")

    print("Loading class names...")
    idx_to_class = load_class_names_from_coco(ann_path)
    print(f"Classes: {idx_to_class}")

    print("Loading model (this may take a moment)...")
    model = load_model(model_path, num_classes=NUM_CLASSES, device=device)
    print("Model loaded.")

    print("Running inference...")
    result = run_inference(model, sample_image, idx_to_class=idx_to_class, device=device)

    print("\nInference result:")
    print(json.dumps(result, indent=2))

    # Prepare output folder
    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)

    # Save result to file
    out_json = os.path.join(outputs_dir, 'last_inference_result.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    out_json = os.path.abspath(out_json)
    print(f"Saved result to: {out_json}")

    # Save annotated image (load image with PIL to avoid OpenCV path encoding issues)
    annotated_path = os.path.join(outputs_dir, 'last_inference_annotated.jpg')
    try:
        pil_img = Image.open(sample_image).convert('RGB')
        img_np = np.array(pil_img)
        # Avoid cv2.imwrite unicode path issues by using output_path=None and saving with PIL
        annotated_img = draw_detections_opencv(img_np, result, output_path=None)
        Image.fromarray(annotated_img).save(annotated_path)
        annotated_path = os.path.abspath(annotated_path)
        print(f"Annotated image saved to: {annotated_path}")
        if args.open and sys.platform.startswith('win'):
            try:
                os.startfile(annotated_path)
            except Exception as e:
                print(f"Failed to open image: {e}")
    except Exception as e:
        print(f"Failed to save annotated image: {e}")


if __name__ == '__main__':
    main()
