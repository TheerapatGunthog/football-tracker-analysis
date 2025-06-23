# %%
import os
import supervision as sv
import torch
import numpy as np
import umap
from sports.common.team import TeamClassifier
from sklearn.cluster import KMeans
from more_itertools import chunked
from transformers import AutoProcessor, SiglipVisionModel
from tqdm import tqdm
from pathlib import Path
from inference import get_model
from dotenv import load_dotenv

load_dotenv()


# %%
# Load Models from Roboflow
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc-nzxsd/1"
PLAYER_DETECTION_MODEL = get_model(PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)


# %%
# Load Models from Hugging Face
MODEL_ID = "google/siglip-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(MODEL_ID).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)


# %%
# Define paths
PROJECT_PATH = Path("/home/whilebell/Code/football-tracker-analysis/")
SOURCE_VIDEO_PATH = PROJECT_PATH / "data/testing_video/121364_0.mp4"
OUTPUT_VIDEO_PATH = PROJECT_PATH / "data/output_videos/121364_0.mp4"

# %%
STRIDE = 30
PLAYER_ID = 2


def extract_crop(source_video_path: str):
    # Extracts crops of players from the video source using the player detection model.
    frame_generator = sv.get_video_frames_generator(source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc="collecting crops"):
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += players_crops

    return crops


# %%
# Extract crops from the video source
crops = extract_crop(SOURCE_VIDEO_PATH)
len(crops)

BATCH_SIZE = 32

crops = [sv.cv2_to_pillow(crop) for crop in crops]
batches = chunked(crops, BATCH_SIZE)
data = []
with torch.no_grad():
    for batch in tqdm(batches, desc="embedding extraction"):
        inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
        outputs = EMBEDDINGS_MODEL(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        data.append(embeddings)

data = np.concatenate(data)


# %%
REDUCER = umap.UMAP(n_components=3)
CLUSTERING_MODEL = KMeans(n_clusters=2)

projections = REDUCER.fit_transform(data)
clusters = CLUSTERING_MODEL.fit_predict(projections)


# %%
def resolve_goalkeepers_team_id(
    players_detections: sv.Detections, goalkeeper_detections: sv.Detections
):
    goalkeepers_xy = goalkeeper_detections.get_anchors_coordinates(
        sv.Position.BOTTOM_CENTER
    )
    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    team_0_centroid = players_xy[players_detections.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_detections.class_id == 1].mean(axis=0)

    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)


# %%
# frame generator for the video source
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

crops = extract_crop(SOURCE_VIDEO_PATH)
team_classifier = TeamClassifier(device=DEVICE)
team_classifier.fit(crops)

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(
        ["#fabd2f", "#83a598", "#fb4934"]
    ),  # Yellow, Blue, Red
    thickness=2,
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(
        ["#fabd2f", "#83a598", "#fb4934"]
    ),  # Yellow, Blue, Red
    text_color=sv.Color.from_hex("#ebdbb2"),  # Foreground
    text_position=sv.Position.BOTTOM_CENTER,
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#fabd2f"), base=20, height=20  # Yellow
)

tracker = sv.ByteTrack()
tracker.reset()

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(OUTPUT_VIDEO_PATH, video_info=video_info)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        results = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(results)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=5)
        all_other_detections = detections[detections.class_id != BALL_ID]
        all_other_detections = all_other_detections.with_nms(
            threshold=0.5, class_agnostic=True
        )
        all_other_detections = tracker.update_with_detections(all_other_detections)

        players_detections = all_other_detections[
            all_other_detections.class_id == PLAYER_ID
        ]
        goalkeepers_detections = all_other_detections[
            all_other_detections.class_id == GOALKEEPER_ID
        ]
        referees_detections = all_other_detections[
            all_other_detections.class_id == REFEREE_ID
        ]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)

        goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
            players_detections, goalkeepers_detections
        )

        referees_detections.class_id -= 1

        all_detections = sv.Detections.merge(
            [players_detections, goalkeepers_detections, referees_detections]
        )
        all_detections.class_id = all_detections.class_id.astype(int)
        if all_detections.tracker_id is not None:
            all_detections.tracker_id = all_detections.tracker_id.astype(int)

        labels = [f"{tracker_id}" for tracker_id in all_detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(annotated_frame, all_detections)
        annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame, all_detections, labels=labels
        )

        video_sink.write_frame(annotated_frame)
