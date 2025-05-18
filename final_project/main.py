import os
import torch
import torchvision
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# === CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FG_CLASSES = 6
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
BACKGROUND_CLASS_INDEX_FOR_CM = 0 # Define globally for access in main block

TRAIN_IMG_ROOT = "./Dataset/train_images"
TRAIN_ANN_ROOT = "./Dataset/train_annotations"
VAL_IMG_ROOT_FOR_SUBMISSION = "./Dataset/test"
SUBMISSION_PATH = "./submission.csv"
TRAIN_VAL_SPLIT_RATIO = 0.8

# === LABELS (1-indexed) ===
LABEL_MAP = {
    'crazing': 1,
    'inclusion': 2,
    'patches': 3,
    'pitted_surface': 4,
    'rolled-in_scale': 5,
    'scratches': 6
}
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class NEUDETDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        if ann_dir:
            self.imgs = [
                img for img in self.imgs
                if os.path.exists(os.path.join(ann_dir, img.replace('.jpg', '.xml')))
            ]

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        if self.ann_dir and os.path.exists(os.path.join(self.ann_dir, img_name.replace('.jpg', '.xml'))):
            ann_path = os.path.join(self.ann_dir, img_name.replace('.jpg', '.xml'))
            tree = ET.parse(ann_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                label_name = obj.find('name').text
                if label_name in LABEL_MAP:
                    label = LABEL_MAP[label_name]
                    bbox = obj.find('bndbox')
                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}
        
        if boxes.shape[0] == 0:
            pass

        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            image = F.to_tensor(image)

        return image, target

    def __len__(self):
        return len(self.imgs)

class ToTensorTarget:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_transforms(is_train):
    return ToTensorTarget()


def get_model():
    model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_FG_CLASSES + 1)
    return model.to(DEVICE)


def collate_fn(batch):
    return tuple(zip(*batch))

def calculate_iou(box1, box2):
    if isinstance(box1, np.ndarray):
        box1 = torch.from_numpy(box1)
    if isinstance(box2, np.ndarray):
        box2 = torch.from_numpy(box2)

    x1_inter = torch.max(box1[0], box2[0])
    y1_inter = torch.max(box1[1], box2[1])
    x2_inter = torch.min(box1[2], box2[2])
    y2_inter = torch.min(box1[3], box2[3])

    inter_width = torch.clamp(x2_inter - x1_inter, min=0)
    inter_height = torch.clamp(y2_inter - y1_inter, min=0)
    intersection = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    iou = intersection / (union + 1e-6)
    return iou.item()

def calculate_ap_pr_rc(precisions, recalls):
    if not recalls or not precisions:
        return 0.0, [], []
        
    sorted_indices = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_indices]
    precisions = np.array(precisions)[sorted_indices]

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i+1])

    recall_points = np.linspace(0, 1, 11)
    ap_precisions = []
    for r_point in recall_points:
        idx = np.where(recalls >= r_point)[0]
        if len(idx) == 0:
            ap_precisions.append(0.0)
        else:
            ap_precisions.append(np.max(precisions[idx]))

    ap = np.mean(ap_precisions)
    return ap, precisions.tolist(), recalls.tolist()

def plot_confusion_matrix(y_true, y_pred, class_names_map, save_path="confusion_matrix.png"):
    """
    Computes and plots the confusion matrix.
    Args:
        y_true (list): List of true labels.
        y_pred (list): List of predicted labels.
        class_names_map (dict): Dictionary mapping class indices to class names (e.g., {0: 'Background', 1: 'crazing', ...}).
        save_path (str): Path to save the confusion matrix plot.
    """
    # Ensure all actual labels present in y_true or y_pred are in class_names_map keys for tick labels
    present_labels = sorted(list(set(y_true) | set(y_pred)))
    tick_labels = [class_names_map.get(i, f"Class {i}") for i in present_labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels,
                yticklabels=tick_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"[INFO] Confusion matrix saved to {save_path}")
    except Exception as e:
        print(f"[ERROR] Could not save confusion matrix: {e}")
    plt.close()

def evaluate_model(model, val_loader, device, num_fg_classes, iou_threshold=0.5):
    model.eval()
    
    all_predictions_by_class = {cls_idx: [] for cls_idx in range(1, num_fg_classes + 1)}
    num_gt_by_class = {cls_idx: 0 for cls_idx in range(1, num_fg_classes + 1)}
    
    # Lists for confusion matrix
    y_true_for_cm = []
    y_pred_for_cm = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                gt_boxes_img = targets[i]['boxes'].cpu()
                gt_labels_img = targets[i]['labels'].cpu()

                for cls_idx in range(1, num_fg_classes + 1):
                    num_gt_by_class[cls_idx] += sum(gt_labels_img == cls_idx).item()

                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                pred_labels = output['labels'].cpu()
                
                sorted_indices = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[sorted_indices]
                pred_labels = pred_labels[sorted_indices]
                pred_scores = pred_scores[sorted_indices]

                gt_matched_in_image = [False] * len(gt_boxes_img)

                for j in range(len(pred_boxes)):
                    pred_box = pred_boxes[j]
                    pred_label = pred_labels[j].item()
                    pred_score = pred_scores[j].item()

                    if pred_label == 0 or pred_label > num_fg_classes:
                        continue

                    max_iou_for_pred = 0.0
                    best_gt_match_idx = -1

                    for k in range(len(gt_boxes_img)):
                        if gt_labels_img[k].item() == pred_label and not gt_matched_in_image[k]:
                            iou = calculate_iou(pred_box, gt_boxes_img[k])
                            if iou > max_iou_for_pred:
                                max_iou_for_pred = iou
                                best_gt_match_idx = k
                    
                    is_tp = False
                    if max_iou_for_pred >= iou_threshold and best_gt_match_idx != -1:
                        if not gt_matched_in_image[best_gt_match_idx]:
                           is_tp = True
                           gt_matched_in_image[best_gt_match_idx] = True
                    
                    all_predictions_by_class[pred_label].append({'score': pred_score, 'tp': is_tp})

                    # Update confusion matrix lists
                    if is_tp:
                        y_true_for_cm.append(gt_labels_img[best_gt_match_idx].item())
                        y_pred_for_cm.append(pred_label)
    
    aps = {}
    total_aps = []
    print("\n--- Per-Class AP @ IoU={} ---".format(iou_threshold))
    for cls_idx in range(1, num_fg_classes + 1):
        class_preds = sorted(all_predictions_by_class[cls_idx], key=lambda x: x['score'], reverse=True)
        
        tp_count = 0
        fp_count = 0
        precisions = []
        recalls = []
        
        num_total_gt_for_class = num_gt_by_class[cls_idx]

        if num_total_gt_for_class == 0:
            ap = 0.0 if len(class_preds) > 0 else 1.0 # AP is 1.0 if no GT and no Preds, 0.0 if no GT but has Preds (FPs)
        else:
            for pred in class_preds:
                if pred['tp']:
                    tp_count += 1
                else:
                    fp_count += 1
                
                current_precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
                current_recall = tp_count / num_total_gt_for_class if num_total_gt_for_class > 0 else 0.0
                precisions.append(current_precision)
                recalls.append(current_recall)

            ap, _, _ = calculate_ap_pr_rc(precisions, recalls)

        class_name = REV_LABEL_MAP.get(cls_idx, f"Class {cls_idx}")
        print(f"{class_name}: {ap:.4f}")
        aps[class_name] = ap
        
        if num_total_gt_for_class > 0:
            total_aps.append(ap)

    mAP = np.mean(total_aps) if total_aps else 0.0
    print(f"-----------------------------mAP@{iou_threshold}: {mAP:.4f}")

    return mAP, aps, y_true_for_cm, y_pred_for_cm

def train_model(model, train_loader, val_loader, epochs=10, patience=5, save_path="best_model.pth"):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    best_val_mAP = -float('inf')
    patience_counter = 0
    epoch_losses = []
    val_mAPs = []
    val_epoch_losses = [] # ADDED: For validation losses

    for epoch in range(epochs):
        model.train()
        total_train_loss_epoch = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [T]")
        for images, targets in progress_bar:
            images = [img.to(DEVICE) for img in images]
            targets_dev = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets_dev)
            loss = sum(l for l in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss_epoch += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_train_loss_epoch = total_train_loss_epoch / len(train_loader)
        epoch_losses.append(avg_train_loss_epoch)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss_epoch:.4f}")

        if val_loader:
            # evaluate_model sets model.eval() internally
            current_val_mAP, _val_aps, _, _ = evaluate_model(model, val_loader, DEVICE, NUM_FG_CLASSES, IOU_THRESHOLD)
            val_mAPs.append(current_val_mAP)
            print(f"Epoch {epoch+1} Validation mAP@{IOU_THRESHOLD}: {current_val_mAP:.4f}")

            # --- ADDED: Validation Loss Calculation ---
            model.train() # Set to train mode temporarily to get losses
            total_val_loss_epoch = 0
            val_loss_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [V Loss]")
            with torch.no_grad(): # Ensure no gradients are computed
                for val_images, val_targets in val_loss_progress_bar:
                    val_images = [img.to(DEVICE) for img in val_images]
                    val_targets_dev = [{k: v.to(DEVICE) for k, v in t.items()} for t in val_targets]
                    val_loss_dict = model(val_images, val_targets_dev)
                    val_loss_batch = sum(l for l in val_loss_dict.values())
                    total_val_loss_epoch += val_loss_batch.item()
                    val_loss_progress_bar.set_postfix(loss=val_loss_batch.item())
            avg_val_loss_epoch = total_val_loss_epoch / len(val_loader)
            val_epoch_losses.append(avg_val_loss_epoch)
            print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss_epoch:.4f}")
            # Model will be set to train() at the start of the next epoch by model.train() call

            if current_val_mAP > best_val_mAP:
                best_val_mAP = current_val_mAP
                torch.save(model.state_dict(), save_path)
                print(f"[INFO] Saved new best model to {save_path} (mAP: {best_val_mAP:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[INFO] Early stopping triggered. No improvement in mAP for {patience} epochs.")
                    break
        else: # No validation loader
            if avg_train_loss_epoch < best_val_mAP: # Repurpose best_val_mAP as best_train_loss
                best_val_mAP = avg_train_loss_epoch
                torch.save(model.state_dict(), save_path)
                print(f"[INFO] Saved new best model to {save_path} (train_loss: {avg_train_loss_epoch:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[INFO] Early stopping triggered. No improvement in train_loss for {patience} epochs.")
                    break
    return epoch_losses, val_mAPs, val_epoch_losses # MODIFIED: Return val_epoch_losses

def plot_training_progress(train_losses, val_epoch_losses, val_metrics, metric_name="mAP"):
    plt.figure(figsize=(12, 5)) # Adjusted figsize back, or make it (18,5) if preferred
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    if val_epoch_losses: # MODIFIED: Plot val_epoch_losses if available
        plt.plot(val_epoch_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss') # MODIFIED: Title
    plt.legend()

    if val_metrics:
        plt.subplot(1, 2, 2)
        plt.plot(val_metrics, label=f'Validation {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'Validation {metric_name} over Epochs')
        plt.legend()
    plt.tight_layout()
    plt.savefig("training_progress.png")
    print("[INFO] Training progress plot saved to training_progress.png")


if __name__ == "__main__":
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Number of foreground classes: {NUM_FG_CLASSES}")
    print(f"[INFO] Label map: {LABEL_MAP}")

    full_train_dataset = NEUDETDataset(TRAIN_IMG_ROOT, TRAIN_ANN_ROOT, transforms=ToTensorTarget())
    print(f"[INFO] Loaded {len(full_train_dataset)} images and annotations from {TRAIN_IMG_ROOT} and {TRAIN_ANN_ROOT}")

    num_total_samples = len(full_train_dataset)
    num_train_samples = int(num_total_samples * TRAIN_VAL_SPLIT_RATIO)
    num_val_samples = num_total_samples - num_train_samples

    if num_train_samples == 0 or num_val_samples == 0:
        print(f"[WARNING] Not enough data to split into training and validation (Train: {num_train_samples}, Val: {num_val_samples}). Training on all {num_total_samples} samples, no validation will be performed.")
        train_dataset_subset = full_train_dataset
        val_dataset_subset = None
    else:
        print(f"[INFO] Splitting data: {num_train_samples} for training, {num_val_samples} for validation.")
        train_dataset_subset, val_dataset_subset = random_split(full_train_dataset, [num_train_samples, num_val_samples])
    
    BATCH_SIZE = 4
    train_loader = DataLoader(train_dataset_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    
    val_loader = None
    if val_dataset_subset and len(val_dataset_subset) > 0:
        val_loader = DataLoader(val_dataset_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
        print(f"[INFO] Created validation loader with {len(val_dataset_subset)} samples.")
    else:
        print("[INFO] No validation loader created (validation subset is empty or not created).")

    model = get_model()
    
    epochs_to_train = 100
    patience_for_early_stopping = 5

    print(f"[INFO] Starting training for {epochs_to_train} epochs...")
    # MODIFIED: Unpack val_epoch_losses
    epoch_losses, val_mAPs, val_epoch_losses = train_model(model, train_loader, val_loader, 
                                         epochs=epochs_to_train, 
                                         patience=patience_for_early_stopping, 
                                         save_path="best_model.pth")

    if epoch_losses:
        # MODIFIED: Pass val_epoch_losses to plotting function
        plot_training_progress(epoch_losses, 
                               val_epoch_losses if val_loader and val_epoch_losses else None, 
                               val_mAPs if val_loader and val_mAPs else None)

    if val_loader:
        print("[INFO] Evaluating the best model on the validation set for final mAP and Confusion Matrix...")
        if os.path.exists("best_model.pth"):
            model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
            model.to(DEVICE)
            final_mAP, final_aps_per_class, y_true_cm_final, y_pred_cm_final = evaluate_model(model, val_loader, DEVICE, NUM_FG_CLASSES, IOU_THRESHOLD)
            print(f"[INFO] Final mAP@{IOU_THRESHOLD} on validation set (best model): {final_mAP:.4f}")

            # Prepare class names for confusion matrix plotting, including background
            CLASS_NAMES_FOR_CM = {BACKGROUND_CLASS_INDEX_FOR_CM: "Background"}
            for k, v in REV_LABEL_MAP.items(): # REV_LABEL_MAP maps 1-6 to names
                CLASS_NAMES_FOR_CM[k] = v
            
            if y_true_cm_final and y_pred_cm_final:
                plot_confusion_matrix(y_true_cm_final, y_pred_cm_final, CLASS_NAMES_FOR_CM, save_path="confusion_matrix_final.png")
            else:
                print("[INFO] No data collected for confusion matrix during final evaluation.")

        else:
            print("[WARNING] best_model.pth not found. Skipping final evaluation and confusion matrix.")
    else:
        print("[INFO] No validation loader was used, skipping final evaluation and confusion matrix.")

    print("[INFO] Training script finished.")
    print(f"[INFO] To generate predictions on a test set (e.g., from {VAL_IMG_ROOT_FOR_SUBMISSION}), run:")
    print(f"python predict.py --model_path best_model.pth --output_csv {SUBMISSION_PATH}")
