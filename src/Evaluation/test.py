y_pred_list = []
y_true_list = []

model.eval()
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred_list.extend(predicted.cpu().numpy())
        y_true_list.extend(labels.cpu().numpy())
try:
    class_names = list(test_loader.dataset.subset.dataset.class_to_idx.keys())
except:
    class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
                    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
                    'River', 'SeaLake']

print("\nClassification Report:")
print(classification_report(y_true_list, y_pred_list, target_names=class_names))

cm = confusion_matrix(y_true_list, y_pred_list)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted ')
plt.ylabel('Actual ')
plt.title('Confusion Matrix')
plt.show()



from torchmetrics.classification import MulticlassROC  
model.eval()
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Evaluating"):
        inputs = inputs.to(device)
        labels = labels.to(device)     
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        y_probs_list.append(probs.cpu())
        y_true_list.append(labels.cpu())
y_probs = torch.cat(y_probs_list)
y_true = torch.cat(y_true_list)
mc_roc = MulticlassROC(num_classes=10, thresholds=None)
mc_roc.update(y_probs, y_true)
fig, ax = mc_roc.plot(score=True) 
plt.title('Multi-class ROC Curve (PyTorch Native)')
plt.show()






CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

def predict_single_image(image_path, model, device):
    with rasterio.open(image_path) as src:
        image_np = src.read()
    image_tensor = torch.from_numpy(image_np).float()
    test_transform = MultiSpectralTestTransform(mean=MS_MEAN, std=MS_STD, resize_to=RESIZE_TO)
    image_tensor = test_transform(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    model.eval()
    with torch.no_grad(): 
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)
        confidence = probabilities[0][predicted_idx].item() * 100
        predicted_class = CLASSES[predicted_idx.item()]
    return predicted_class, confidence


test_image_path = r"D:\OneDrive - Computer and Information Technology (Menofia University)\Desktop\test\EuroSAT_MS\River\River_10.tif"

cls_name, conf = predict_single_image(test_image_path, model, device)
print(f"Prediction: {cls_name}")
print(f"Confidence: {conf:.2f}%")

