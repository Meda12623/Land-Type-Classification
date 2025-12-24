import streamlit as st
import torch
import torch.nn.functional as F
import rasterio
from rasterio.plot import reshape_as_image
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import models
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class PrependResNet50(nn.Module):
    def __init__(self, num_classes=10, n_pre_convs=1, deep_head_dims=[1024,512], p_dropout=0.4, pretrained=True):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.base = base

        old_conv = self.base.conv1
        self.base.conv1 = nn.Conv2d(64, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                    stride=old_conv.stride, padding=old_conv.padding, bias=False)

        layers = []
        in_ch = 13
        out_ch = 64
        for i in range(n_pre_convs):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.prepend = nn.Sequential(*layers)
    
        self.prepend.apply(init_weights_kaiming)

        in_feat = self.base.fc.in_features
        head_layers = []
        last = in_feat
        for h in deep_head_dims:
            head_layers += [ nn.Linear(last, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(p_dropout) ]
            last = h
        head_layers += [ nn.Linear(last, num_classes) ]
        self.base.fc = nn.Sequential(*head_layers)

        self.base.fc.apply(init_weights_kaiming)

    def forward(self, x):
        x = self.prepend(x)          
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.fc(x)
        return x

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

RESIZE_TO = (96, 96)

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

def manual_resize(image: torch.Tensor, new_height: int, new_width: int) -> torch.Tensor:
    dtype = image.dtype
    device = image.device
    image = image.unsqueeze(0).to(dtype=dtype, device=device)
    resized = F.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)
    return resized.squeeze(0)

class MultiSpectralTestTransform:
    def __init__(self, mean, std, resize_to):
        self.mean = mean
        self.std = std
        self.resize_to = resize_to

    def __call__(self, img):
        img = manual_resize(img, self.resize_to[0], self.resize_to[1])
        img = (img - self.mean[:, None, None]) / self.std[:, None, None]
        return img.to(device)

@st.cache_resource
def load_model():
    model = PrependResNet50(num_classes=10).to(device)
    model_path = "best_model.pth"

    temp = torch.load(model_path, map_location=device)
    model.load_state_dict(temp)
    return model

def get_rgb_visualization(image_data):
    try:
        rgb_bands = image_data[[3, 2, 1], :, :]
        rgb_image = reshape_as_image(rgb_bands)

        p2, p98 = np.percentile(rgb_image, (2, 98))
        rgb_image = np.clip(rgb_image, p2, p98)
        
        rgb_image = (rgb_image - p2) / (p98 - p2) * 255
        
        return Image.fromarray(rgb_image.astype('uint8'))
    except Exception as e:
        print(f"Visualization Error: {e}")
        return None

st.set_page_config(page_title="Land Type Classifier", layout='wide')

st.title("Sentinel-2 Land Classification")
st.markdown("### Satellite Imagery Analysis")
st.markdown("---")


with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Select Sentinel-2 Image (.tif)", type=["tif", "tiff"])
    st.info("Ensure the image contains 13 channel.")


if uploaded_file is not None:
    try:
        with rasterio.open(uploaded_file) as src:
            image_np = src.read()
            meta = src.meta
            
        
        st.subheader("RGB Image")
        rgb_img = get_rgb_visualization(image_np)
        if rgb_img:
            st.image(rgb_img, caption="RGB Composite", width=600)
        else:
            st.warning("Unable to generate RGB preview.")

        st.subheader("Analysis Results")

        if st.button("Analyze The Image"):
            with st.spinner('Processing...'):
                model = load_model()

                if model is not None:
                    img_tensor = torch.from_numpy(image_np).float()
                    transform = MultiSpectralTestTransform(MS_MEAN, MS_STD, RESIZE_TO)
                    img_tensor = transform(img_tensor)
                    img_tensor = img_tensor.unsqueeze(0)

                    model.eval()
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = F.softmax(outputs, dim=1)
                        conf, pred_idx = torch.max(probs, 1)

                    pred_class = CLASSES[pred_idx.item()]
                    confidence = conf.item()
                    
                    st.success(f"**Class:** {pred_class}")
                    st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
                    st.progress(confidence)

                    # with st.expander("View All Probabilities"):
                    #     p = {}
                    #     for i, class_name in enumerate(CLASSES):
                    #         p[class_name] = f"{probs[0][i].item()*100:.2f}%"
                    #     st.table(p)

                    with st.expander("Advanced more obtions"):
                        col1, col2 = st.columns(2)
                        with col1:
                            data = []
                            for i, class_name in enumerate(CLASSES):
                                prob_value = probs[0][i].item()
                                data.append({
                                    "Land Type": class_name, 
                                    "Probability": f"{prob_value*100:.2f}%",
                                    "Value": prob_value
                                })
                            
                            df = pd.DataFrame(data)
                            df = df.sort_values(by="Value", ascending=False)
                            df_display = df[["Land Type", "Probability"]]
                            st.table(df_display)

                        with col2:
                            st.write("**Probability Distribution:**")
                            probs_np = probs.cpu().numpy()[0]
                            st.bar_chart({CLASSES[i]: probs_np[i] for i in range(len(CLASSES))})

                else:
                    st.error("Model not loaded. Check if 'best_model.pth' exists.")

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.warning("#### Uploading the Image")