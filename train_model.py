from ultralytics import YOLO

# Charger le modèle pré-entraîné
model = YOLO("yolov8n.pt")

# Entraîner le modèle
results = model.train(
    data="mon_dataset/FallDown.v3-person_bed.yolov8/data.yaml",
    epochs=5,
    imgsz=640,
    batch=16,  # Ajustez selon votre mémoire GPU/CPU
    name="falldown_detection",  # Nom du dossier de résultats
    project="runs/detect",  # Dossier principal des résultats
    device="cpu",  # Utilisation du CPU (pas de GPU CUDA disponible)
    patience=50,  # Early stopping
    save=True,
    plots=True,
)

print("\n✅ Entraînement terminé!")
print(f"📊 Résultats dans: runs/detect/falldown_detection")
