import pandas as pd
import PIL.Image
from pathlib import Path
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

square_tensor_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

class MisoValDataset(Dataset):
    def __init__(self, path, miso_count = 30000, nmiso_count = 30000, image_transform = square_tensor_transform, shuffule = False, shuffle_seed = 42,deciding_label = "indian_label"):
        super().__init__()
        self.image_transform = image_transform

        df = pd.read_csv(Path(path) / "dev.csv")
        directory = Path(path)
        
        self.data = []
        
        miso_data = []
        nmiso_data = []
        
        # Iterate through jpg files
        for file in tqdm(directory.glob('*.jpg'),desc="Files Left"):
            
            # 1. Get the image_id (filename without .jpg)
            img_id = int(file.stem)
            
            # 2. Open and transform the image
            img_pil = PIL.Image.open(file).convert('RGB')
            img_tensor = self.image_transform(img_pil)
            
            # 3. Get the transcription matching the image_id
            transcription = df.loc[df['image_id'] == img_id]['transcriptions'].iloc[0]
            indian_label = df.loc[df['image_id'] == img_id]['indian_labels'].iloc[0]
            chinese_label = df.loc[df['image_id'] == img_id]['chinese_labels'].iloc[0]
            
            indian_label = 1 if indian_label == "misogyny" else 0
            chinese_label = 1 if chinese_label == "misogyny" else 0

            data_item = {
                "image_id": img_id,
                "img": img_tensor,
                "transcription": transcription,
                "indian_label": indian_label,
                "chinese_label": chinese_label,
                }
            
            if data_item[deciding_label]: miso_data.append(data_item)
            else: nmiso_data.append(data_item)

        if shuffule:
            random.seed(42)
            random.shuffle(miso_data)
            random.shuffle(nmiso_data)

        for i in range(min(nmiso_count,len(nmiso_data))): self.data.append(nmiso_data[i])
        for i in range(min(miso_count,len(miso_data))): self.data.append(miso_data[i])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class MisoTrainDataset(Dataset):
    def __init__(self, path, miso_count = 30000, nmiso_count = 30000, image_transform = square_tensor_transform, shuffule = False, shuffle_seed = 42,deciding_label = "indian_label"):
        super().__init__()
        self.image_transform = image_transform

        df = pd.read_csv(Path(path) / "train.csv")
        directory = Path(path)
        
        self.data = []
        
        miso_data = []
        nmiso_data = []
        
        # Iterate through jpg files
        for file in tqdm(directory.glob('*.jpg'),desc="Files Left"):
            
            # 1. Get the image_id (filename without .jpg)
            img_id = int(file.stem)
            
            # 2. Open and transform the image
            img_pil = PIL.Image.open(file).convert('RGB')
            img_tensor = self.image_transform(img_pil)
            
            # 3. Get the transcription matching the image_id
            transcription = df.loc[df['image_id'] == img_id]['transcriptions'].iloc[0]
            indian_label = df.loc[df['image_id'] == img_id]['indian_labels'].iloc[0]
            chinese_label = df.loc[df['image_id'] == img_id]['chinese_labels'].iloc[0]
            
            indian_label = 1 if indian_label == "misogyny" else 0
            chinese_label = 1 if chinese_label == "misogyny" else 0

            data_item = {
                "image_id": img_id,
                "img": img_tensor,
                "transcription": transcription,
                "indian_label": indian_label,
                "chinese_label": chinese_label,
                }
            
            if data_item[deciding_label]: miso_data.append(data_item)
            else: nmiso_data.append(data_item)

        if shuffule:
            random.seed(42)
            random.shuffle(miso_data)
            random.shuffle(nmiso_data)

        for i in range(min(nmiso_count,len(nmiso_data))): self.data.append(nmiso_data[i])
        for i in range(min(miso_count,len(miso_data))): self.data.append(miso_data[i])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
            
