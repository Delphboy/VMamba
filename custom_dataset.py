import json
import os
import json
from torch.utils.data import Dataset
from PIL import Image

class CaptioningDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        captions_file: str,
        transforms,
        split: str = "train",
    ):
        self.root_dir = root_dir
        self.captions_file = captions_file
        self.transforms = transforms

        assert split in ["train", "val", "test"], f"BAD SPLIT must be train, val or test. Received: {split}"
        self.split = split

        with open(self.captions_file, "r") as f:
            self.captions_file_data = json.load(f)

        self.image_locations = []
        self.cocoids = []

        for image_data in self.captions_file_data["images"]:
            if image_data["split"] == "restval":
                image_data["split"] = "train"

            if image_data["split"] == self.split:
                img_path = os.path.join(
                    self.root_dir,
                    f"{image_data['filepath']}",
                    f"{image_data['filename']}"
                )
                cocoid = image_data['cocoid']

                self.image_locations.append(img_path)
                self.cocoids.append(cocoid)

    def __getitem__(self, index):
        img_path = self.image_locations[index]
        cocoid = self.cocoids[index]

        image = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)

        return image, cocoid


    def __len__(self): return len(self.image_locations)

