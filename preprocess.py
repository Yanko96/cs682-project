import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import get_worker_info

# Configs
SPLIT = 'train'
VITON_ROOT = f'/PATH/TO/VITON/RAW_DATA/{SPLIT}'
SAVE_ROOT = f'/PATH/TO/VITON/PREPROCESSED_DATA/{SPLIT}'

image_dir = os.path.join(VITON_ROOT, 'image')
cloth_dir = os.path.join(VITON_ROOT, 'cloth')
pose_vis_dir = os.path.join(VITON_ROOT, 'openpose_img')
pose_dir = os.path.join(VITON_ROOT, 'openpose_json')
parse_pose_dir = os.path.join(VITON_ROOT, 'image-densepose')
agnostic_dir = os.path.join(VITON_ROOT, 'agnostic-v3.2')
parse_agnostic_dir = os.path.join(VITON_ROOT, 'image-parse-agnostic-v3.2')
parse_dir = os.path.join(VITON_ROOT, 'image-parse-v3')
pair_path = os.path.join(VITON_ROOT, '../', f'{SPLIT}_pairs.txt')

os.makedirs(os.path.join(SAVE_ROOT, 'pose'), exist_ok=True)
os.makedirs(os.path.join(SAVE_ROOT, 'pose_map'), exist_ok=True)
os.makedirs(os.path.join(SAVE_ROOT, 'parse_pose'), exist_ok=True)
os.makedirs(os.path.join(SAVE_ROOT, 'cloth'), exist_ok=True)
os.makedirs(os.path.join(SAVE_ROOT, 'person'), exist_ok=True)
os.makedirs(os.path.join(SAVE_ROOT, 'agnostic'), exist_ok=True)
os.makedirs(os.path.join(SAVE_ROOT, 'parse_agnostic'), exist_ok=True)
os.makedirs(os.path.join(SAVE_ROOT, 'parse'), exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((512, 384)),
    transforms.ToTensor()
])

COCO_18_FROM_BODY_25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11, 1]

def extract_coco18_from_body25(keypoints_25):
    keypoints_18 = []
    for idx in COCO_18_FROM_BODY_25:
        x = keypoints_25[idx * 3]
        y = keypoints_25[idx * 3 + 1]
        v = keypoints_25[idx * 3 + 2]
        keypoints_18.extend([x, y, v])
    return keypoints_18

def load_pose_map(pose_json_path, H=512, W=384):
    if not os.path.exists(pose_json_path):
        return None
    with open(pose_json_path, 'r') as f:
        pose_data = json.load(f)
    pose_map = np.zeros((18, H, W))
    if len(pose_data['people']) == 0:
        return None
    keypoints_25 = pose_data['people'][0]['pose_keypoints_2d']
    keypoints = extract_coco18_from_body25(keypoints_25)
    for i in range(18):
        x = keypoints[i * 3]
        y = keypoints[i * 3 + 1]
        v = keypoints[i * 3 + 2]
        if v > 0:
            x = int(x * W / 768)
            y = int(y * H / 1024)
            if 0 <= x < W and 0 <= y < H:
                pose_map[i, y, x] = 1
    return torch.from_numpy(pose_map).float()

def process_pair(pair):
    person_img, cloth_img = pair
    name = person_img.replace('.jpg', '')

    pose_vis_path = os.path.join(pose_vis_dir, f'{name}_rendered.png')
    if not os.path.exists(pose_vis_path):
        return None
    pose_img = Image.open(pose_vis_path).convert('RGB')
    pose_tensor = transform(pose_img)

    person_path = os.path.join(image_dir, person_img)
    # cloth_path = os.path.join(cloth_dir, cloth_img)
    cloth_path = os.path.join(cloth_dir, person_img)
    agnostic_path = os.path.join(agnostic_dir, person_img)
    parse_agnostic_path = os.path.join(parse_agnostic_dir, person_img).replace('.jpg', '.png')
    parse_pose_path = os.path.join(parse_pose_dir, person_img)
    parse_path = os.path.join(parse_dir, person_img).replace('.jpg', '.png')
    if not all(map(os.path.exists, [person_path, cloth_path, agnostic_path, parse_agnostic_path, parse_pose_path, parse_path])):
        return None

    person = Image.open(person_path).convert('RGB')
    cloth = Image.open(cloth_path).convert('RGB')
    agnostic = Image.open(agnostic_path).convert('RGB')
    parse_agnostic = Image.open(parse_agnostic_path).convert('RGB')
    parse_pose = Image.open(parse_pose_path).convert('RGB')
    parse = Image.open(parse_path).convert('RGB')
    person_tensor = transform(person)
    cloth_tensor = transform(cloth)
    agnostic_tensor = transform(agnostic)
    parse_agnostic_tensor = transform(parse_agnostic)
    parse_pose_tensor = transform(parse_pose)
    parse_tensor = transform(parse)

    pose_json_path = os.path.join(pose_dir, person_img.replace('.jpg', '_keypoints.json'))
    pose_map = load_pose_map(pose_json_path)
    if pose_map is None:
        return None

    torch.save(person_tensor, os.path.join(SAVE_ROOT, 'person', f'{name}.pt'))
    torch.save(cloth_tensor, os.path.join(SAVE_ROOT, 'cloth', f'{name}.pt'))
    torch.save(pose_tensor,  os.path.join(SAVE_ROOT, 'pose', f'{name}.pt'))
    torch.save(pose_map,     os.path.join(SAVE_ROOT, 'pose_map', f'{name}.pt'))
    torch.save(agnostic_tensor, os.path.join(SAVE_ROOT, 'agnostic', f'{name}.pt'))
    torch.save(parse_agnostic_tensor, os.path.join(SAVE_ROOT, 'parse_agnostic', f'{name}.pt'))
    torch.save(parse_pose_tensor, os.path.join(SAVE_ROOT, 'parse_pose', f'{name}.pt'))
    torch.save(parse_tensor, os.path.join(SAVE_ROOT, 'parse', f'{name}.pt'))
    return True

def preprocess():
    with open(pair_path, 'r') as f:
        pairs = [line.strip().split() for line in f.readlines()]

    from torch.utils.data import DataLoader
    from multiprocessing import cpu_count

    from torch.utils.data import Dataset
    class PairDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __getitem__(self, index):
            return self.data[index]
        def __len__(self):
            return len(self.data)

    dataset = PairDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=cpu_count(), collate_fn=lambda x: x[0])

    valid = 0
    for pair in tqdm(dataloader):
        result = process_pair(pair)
        if result:
            valid += 1

    print(f"Preprocessing is complete, number of valid samples: {valid} / {len(pairs)}")

if __name__ == '__main__':
    preprocess()
