import os
import torch
import torchvision
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn ## just for example
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor ## just for example

# === CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 6
CONF_THRESHOLD = 0.5

TRAIN_IMG_ROOT = "./train_images"
TRAIN_ANN_ROOT = "./train_annotations"
VAL_IMG_ROOT = "./test"
SUBMISSION_PATH = "./submission.csv"

# === LABELS ===
LABEL_MAP = {
    'crazing': 0,
    'inclusion': 1,
    'patches': 2,
    'pitted_surface': 3,
    'rolled-in_scale': 4,
    'scratches': 5
}
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class NEUDETDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name.replace('.jpg', '.xml'))
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(LABEL_MAP[label])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}

        if self.transforms:
            image = self.transforms(image)
        else:
            image = F.to_tensor(image)

        return image, target

    def __len__(self):
        return len(self.imgs)


def get_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True) # for example, we use fastercnn model here
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model.to(DEVICE)


def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(model, train_loader, epochs=10):
    model.train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

def predict_and_export(model, val_dir, save_csv_path, save_img_dir="output"):
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # import os
    # from PIL import Image
    # import pandas as pd
    # from tqdm import tqdm
    # from torchvision.transforms import functional as F

    os.makedirs(save_img_dir, exist_ok=True)
    model.eval()
    rows = []

    val_images = sorted([f for f in os.listdir(val_dir) if f.lower().endswith('.jpg')])

    for img_file in tqdm(val_images, desc="Predicting"):
        img_path = os.path.join(val_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        image_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image_tensor)[0]

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        cf_list, xmin_list, ymin_list, xmax_list, ymax_list = [], [], [], [], []

        # Plot image
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = map(int, box.tolist())
            cf_list.append(f"{score:.2f}")
            xmin_list.append(str(xmin))
            ymin_list.append(str(ymin))
            xmax_list.append(str(xmax))
            ymax_list.append(str(ymax))

            # Draw box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f"{REV_LABEL_MAP[label.item()]} {score:.2f}",
                    color='red', fontsize=8, weight='bold')

        # Save image
        fig.savefig(os.path.join(save_img_dir, img_file), bbox_inches='tight', dpi=150)
        plt.close(fig)

        if len(cf_list) == len(xmin_list) == len(ymin_list) == len(xmax_list) == len(ymax_list) and len(cf_list) > 0:
            rows.append({
                "ID": img_file,
                "label": REV_LABEL_MAP[labels[0].item()] if len(labels) > 0 else "none",
                "cf": " ".join(cf_list),
                "xmin": " ".join(xmin_list),
                "ymin": " ".join(ymin_list),
                "xmax": " ".join(xmax_list),
                "ymax": " ".join(ymax_list),
            })
        else:
            # For images with no detections, fill with placeholders
            rows.append({
                "ID": img_file,
                "label": "none",
                "cf": "0",
                "xmin": "0",
                "ymin": "0",
                "xmax": "0",
                "ymax": "0",
            })

    df = pd.DataFrame(rows)
    df.fillna("none", inplace=True)
    df.to_csv(save_csv_path, index=False)
    print(f"[INFO] Saved CSV to {save_csv_path}")
    print(f"[INFO] Saved annotated images to {save_img_dir}/")

if __name__ == "__main__":
    # Train (optional)
    train_dataset = NEUDETDataset(TRAIN_IMG_ROOT, TRAIN_ANN_ROOT)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = get_model()
    train_model(model, train_loader, epochs=100)

    # Predict and save CSV
    predict_and_export(model, VAL_IMG_ROOT, SUBMISSION_PATH, save_img_dir="output")
