ROOT_DIR = r"D:\OneDrive - Computer and Information Technology (Menofia University)\Desktop\test\EuroSAT_MS"
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 1

RESIZE_TO = (96, 96)

def manual_resize(image: torch.Tensor, new_height: int, new_width: int) -> torch.Tensor:
    dtype = image.dtype
    device = image.device
    image = image.unsqueeze(0).to(dtype=dtype, device=device)
    resized = F.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)
    return resized.squeeze(0)


class EuroSATMSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform

        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self._load_samples(root_dir, classes)

    def _load_samples(self, root_dir, classes):
        for cls_name in classes:
            cls_idx = self.class_to_idx[cls_name]
            cls_folder = os.path.join(root_dir, cls_name)
            img_paths = glob.glob(os.path.join(cls_folder, "*.tif"))
            self.paths.extend(img_paths)
            self.labels.extend([cls_idx] * len(img_paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]

        with rasterio.open(img_path) as src:
            image_np = src.read()         

        image_tensor = torch.from_numpy(image_np).float()

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label



MS_MEAN = torch.tensor([
    1353.7289, 1117.2061, 1041.8864, 946.5517, 1199.1844, 2003.0060,
    2374.0132, 2301.2263, 732.1810, 12.0996, 1820.6929, 1118.2050,
    2599.7854
])

MS_STD = torch.tensor([
    245.2682, 333.4232, 395.2124, 594.4780, 567.0257, 861.0189,
    1086.9409, 1118.3157, 403.8531, 4.7293, 1002.5690, 760.5990,
    1231.6958
])



class MultiSpectralTrainTransformAdvanced:
    def __init__(self, mean, std, resize_to, crop_scale=(1.0, 2.0)):
        self.mean = mean
        self.std = std
        self.resize_to = resize_to
        self.crop_scale = crop_scale

    def __call__(self, img):
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[1])

        k = random.choice([0,1,2,3])
        img = torch.rot90(img, k, dims=[1,2])

        C, H, W = img.shape
        scale = random.uniform(*self.crop_scale)
        new_H, new_W = int(H * scale), int(W * scale)
        img = manual_resize(img, new_H, new_W)

        top = random.randint(0, max(0, new_H - self.resize_to[0]))
        left = random.randint(0, max(0, new_W - self.resize_to[1]))
        img = img[:, top:top + self.resize_to[0], left:left + self.resize_to[1]]

        img = manual_resize(img, self.resize_to[0], self.resize_to[1])

        img = (img - self.mean[:, None, None]) / self.std[:, None, None]

        return img


# Test Transform
class MultiSpectralTestTransform:
    def __init__(self, mean, std, resize_to):
        self.mean = mean
        self.std = std
        self.resize_to = resize_to

    def __call__(self, img):
        img = manual_resize(img, self.resize_to[0], self.resize_to[1])
        img = (img - self.mean[:, None, None]) / self.std[:, None, None]
        return img



class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_and_split_eurosat(root_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")

    train_transform = MultiSpectralTrainTransformAdvanced(MS_MEAN, MS_STD, RESIZE_TO)
    test_transform  = MultiSpectralTestTransform(MS_MEAN, MS_STD, RESIZE_TO)

    full_dataset = EuroSATMSDataset(root_dir, transform=None)

    total = len(full_dataset)
    train_size = int(train_ratio * total)
    val_size   = int(val_ratio * total)
    test_size  = total - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = SubsetWithTransform(train_subset, transform=train_transform)
    val_dataset   = SubsetWithTransform(val_subset,   transform=test_transform)
    test_dataset  = SubsetWithTransform(test_subset,  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = load_and_split_eurosat(ROOT_DIR)
