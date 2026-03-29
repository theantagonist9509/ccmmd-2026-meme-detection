import pandas as pd
import PIL.Image
from pathlib import Path
import torch
import gc
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

square_tensor_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

def create_val_dataset( path, miso_count = 30000, nmiso_count = 30000, image_transform = square_tensor_transform, shuffule = False, shuffle_seed = 42,deciding_label = "indian_label", mode="both"):
    return MisoDataset(path, 
                       label_file="dev.csv", 
                       miso_count=miso_count, 
                       nmiso_count=nmiso_count, 
                       image_transform=image_transform, 
                       shuffule=shuffule, 
                       shuffle_seed=shuffle_seed,
                       deciding_label=deciding_label,
                       mode=mode)

def create_train_dataset( path, miso_count = 30000, nmiso_count = 30000, image_transform = square_tensor_transform, shuffule = False, shuffle_seed = 42,deciding_label = "indian_label", mode="both"):
    return MisoDataset(path, 
                       label_file="train.csv", 
                       miso_count=miso_count, 
                       nmiso_count=nmiso_count, 
                       image_transform=image_transform, 
                       shuffule=shuffule, 
                       shuffle_seed=shuffle_seed,
                       deciding_label=deciding_label,
                       mode=mode)

class MisoDataset(Dataset):
    def __init__(self, path, label_file,miso_count = 30000, nmiso_count = 30000, image_transform = square_tensor_transform, shuffule = False, shuffle_seed = 42,deciding_label = "indian_label", mode="both"):
        super().__init__()
        self.image_transform = image_transform
        self.mode = mode

        df = pd.read_csv(Path(path) / label_file)
        directory = Path(path)

        self.data = []

        miso_data = []
        nmiso_data = []

        # Iterate through jpg files
        for file in tqdm(directory.glob('*.jpg'),desc="Files Left"):

            # 1. Get the image_id (filename without .jpg)
            img_id = int(file.stem)

            # 2. Open and transform the image
            if self.mode in ["image", "both"]:
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
                "indian_label": indian_label,
                "chinese_label": chinese_label,
                }
            if self.mode in ["image", "both"]:
                data_item["img"] = img_tensor
            if self.mode in ["text", "both"]:
                data_item["transcription"] = transcription

            if data_item[deciding_label]: miso_data.append(data_item)
            else: nmiso_data.append(data_item)

        if shuffule:
            random.seed(shuffle_seed)
            random.shuffle(miso_data)
            random.shuffle(nmiso_data)

        for i in range(min(nmiso_count,len(nmiso_data))): self.data.append(nmiso_data[i])
        for i in range(min(miso_count,len(miso_data))): self.data.append(miso_data[i])

        if shuffule:
            random.seed(shuffle_seed)
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class EmbeddedMisoDataset(Dataset):
    def __init__(self, miso_dataset, model, batch_size=32, device=None, **kwargs):
        super().__init__()
        self.data = []
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        loader = DataLoader(miso_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        model.to(device)
        model.eval()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing embeddings"):
                dev_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                outputs = model(dev_batch)

                outputs = outputs.cpu()

                b_size = len(batch["image_id"])
                for i in range(b_size):
                    image_id = batch["image_id"][i].item() if isinstance(batch["image_id"], torch.Tensor) else batch["image_id"][i]
                    indian_label = batch["indian_label"][i].item() if isinstance(batch["indian_label"], torch.Tensor) else batch["indian_label"][i]
                    chinese_label = batch["chinese_label"][i].item() if isinstance(batch["chinese_label"], torch.Tensor) else batch["chinese_label"][i]

                    item = {
                        "image_id": image_id,
                        "embedding": outputs[i],
                        "indian_label": indian_label,
                        "chinese_label": chinese_label,
                    }
                    self.data.append(item)

        model.to('cpu')
        del loader, model, miso_dataset
        if 'batch' in locals():
            del batch, dev_batch, outputs

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
