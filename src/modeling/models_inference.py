from ultralytics import YOLO

model = YOLO(
    "/home/whilebell/Code/football-tracker-analysis/models/soccer_player_detector/train/weights/best.pt"
)

results = model.predict(
    "/home/whilebell/Code/football-tracker-analysis/data/testing_video/08fd33_4.mp4",
    save=True,
    project="/home/whilebell/Code/football-tracker-analysis/models_inference",
    name="player_detector_models",
)

print(results[0])
print("==========================================")
for box in results[0].boxes:
    print(box)
