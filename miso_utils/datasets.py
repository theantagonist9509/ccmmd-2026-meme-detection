import pandas as pd
import PIL.Image
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

class MisoValDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        df = pd.read_csv(Path(path) / "dev.csv")
        directory = Path(path)
        
        # Define the transformation: Resize and convert to Tensor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.data = []
        
        # Iterate through jpg files
        for file in tqdm(directory.glob('*.jpg'),desc="Files Left"):
            
            # 1. Get the image_id (filename without .jpg)
            img_id = int(file.stem)
            
            # 2. Open and transform the image
            img_pil = PIL.Image.open(file).convert('RGB')
            img_tensor = self.transform(img_pil)
            
            # 3. Get the transcription matching the image_id
            transcription = df.loc[df['image_id'] == img_id]['transcriptions'].iloc[0]
            indian_label = df.loc[df['image_id'] == img_id]['indian_labels'].iloc[0]
            chinese_label = df.loc[df['image_id'] == img_id]['chinese_labels'].iloc[0]
            
            indian_label = 1 if indian_label == "misogyny" else 0
            chinese_label = 1 if chinese_label == "misogyny" else 0

            self.data.append({
                "image_id": img_id,
                "img": img_tensor,
                "transcription": transcription,
                "indian_label": indian_label,
                "chinese_label": chinese_label,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class MisoTrainDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        df = pd.read_csv(Path(path) / "train.csv")
        directory = Path(path)
        
        # Define the transformation: Resize and convert to Tensor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.data = []
        
        # Iterate through jpg files
        for file in tqdm(directory.glob('*.jpg'),desc="Files Left"):
            
            # 1. Get the image_id (filename without .jpg)
            img_id = int(file.stem)
            
            # 2. Open and transform the image
            img_pil = PIL.Image.open(file).convert('RGB')
            img_tensor = self.transform(img_pil)
            
            # 3. Get the transcription matching the image_id
            transcription = df.loc[df['image_id'] == img_id]['transcriptions'].iloc[0]
            indian_label = df.loc[df['image_id'] == img_id]['indian_labels'].iloc[0]
            chinese_label = df.loc[df['image_id'] == img_id]['chinese_labels'].iloc[0]
            
            indian_label = 1 if indian_label == "misogyny" else 0
            chinese_label = 1 if chinese_label == "misogyny" else 0

            self.data.append({
                "image_id": img_id,
                "img": img_tensor,
                "transcription": transcription,
                "indian_label": indian_label,
                "chinese_label": chinese_label,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
            
