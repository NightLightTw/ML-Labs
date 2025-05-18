import os
import torch
import torchvision
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FG_CLASSES = 6
CONF_THRESHOLD = 0.5
LABEL_MAP_PREDICT = {
    'crazing': 1,
    'inclusion': 2,
    'patches': 3,
    'pitted_surface': 4,
    'rolled-in_scale': 5,
    'scratches': 6
}
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP_PREDICT.items()}


def get_model_for_prediction(num_fg_classes):
    """
    Helper function to define the model architecture.
    It should be identical to the model architecture used during training.
    """
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # num_fg_classes + 1 because background is class 0
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_fg_classes + 1)
    return model


def predict_and_export(model, test_img_dir, save_csv_path, save_img_dir="output_predictions"):
    """
    Loads a model, performs predictions on images in a directory,
    saves bounding boxes to a CSV, and exports annotated images.
    """
    os.makedirs(save_img_dir, exist_ok=True)
    model.to(DEVICE) # Ensure model is on the correct device
    model.eval()
    rows = []

    test_images = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for img_file in tqdm(test_images, desc="Predicting on test set"):
        img_path = os.path.join(test_img_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not open or read image {img_path}. Skipping. Error: {e}")
            continue
        
        image_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image_tensor)[0]

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        img_display = image.copy()
        fig, ax = plt.subplots(1)
        ax.imshow(img_display)
        ax.axis('off')

        prediction_strings = {
            "cf": [], "xmin": [], "ymin": [], "xmax": [], "ymax": [], "label_names": []
        }
        
        found_detection = False
        for box, score, label_idx in zip(boxes, scores, labels):
            if score < CONF_THRESHOLD:
                continue
            
            if label_idx == 0 or label_idx > NUM_FG_CLASSES:
                # print(f"Skipping label_idx {label_idx} for image {img_file} (score: {score:.2f})")
                continue
            
            found_detection = True
            xmin, ymin, xmax, ymax = map(int, box)
            
            prediction_strings["cf"].append(f"{score:.2f}")
            prediction_strings["xmin"].append(str(xmin))
            prediction_strings["ymin"].append(str(ymin))
            prediction_strings["xmax"].append(str(xmax))
            prediction_strings["ymax"].append(str(ymax))
            
            class_name = REV_LABEL_MAP.get(label_idx, f"Unknown_L{label_idx}")
            prediction_strings["label_names"].append(class_name)

            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"{class_name} {score:.2f}",
                    color='red', fontsize=6, bbox=dict(facecolor='white', alpha=0.5, pad=0))
        
        try:
            fig.savefig(os.path.join(save_img_dir, img_file), bbox_inches='tight', pad_inches=0, dpi=200)
        except Exception as e:
            print(f"Warning: Could not save annotated image {img_file}. Skipping. Error: {e}")
        plt.close(fig)

        if found_detection and prediction_strings["label_names"]:
            main_label = prediction_strings["label_names"][0] 
            rows.append({
                "ID": img_file,
                "label": main_label,
                "cf": " ".join(prediction_strings["cf"]),
                "xmin": " ".join(prediction_strings["xmin"]),
                "ymin": " ".join(prediction_strings["ymin"]),
                "xmax": " ".join(prediction_strings["xmax"]),
                "ymax": " ".join(prediction_strings["ymax"]),
            })
        else:
            rows.append({
                "ID": img_file, "label": "none",
                "cf": "0", "xmin": "0", "ymin": "0", "xmax": "0", "ymax": "0",
            })

    df = pd.DataFrame(rows)
    try:
        df.to_csv(save_csv_path, index=False)
        print(f"[INFO] Saved CSV to {save_csv_path}")
    except Exception as e:
        print(f"[ERROR] Could not save CSV to {save_csv_path}. Error: {e}")
        
    print(f"[INFO] Saved annotated images to {save_img_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict object detections using a trained Faster R-CNN model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth file).")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Path to save the output CSV file.")
    parser.add_argument("--output_img_dir", type=str, default="output", help="Directory to save annotated images.")
    parser.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD, help="Confidence threshold for detections.")
    
    args = parser.parse_args()

    CONF_THRESHOLD = args.conf_threshold
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Confidence threshold: {CONF_THRESHOLD}")


    # Define model structure
    model = get_model_for_prediction(NUM_FG_CLASSES)
    
    # Load the trained model weights
    try:
        print(f"[INFO] Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model from {args.model_path}. Error: {e}")
        exit()

    # Perform prediction and export results
    TEST_IMG_ROOT = "./Dataset/test"
    predict_and_export(model, TEST_IMG_ROOT, args.output_csv, args.output_img_dir)

    print("[INFO] Prediction script finished.")