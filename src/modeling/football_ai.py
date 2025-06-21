# %%
import os
import supervision as sv
import torch
import numpy as np
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
EMBEDDING_MODEL = SiglipVisionModel.from_pretrained(MODEL_ID).to(DEVICE)
EMBEDDING_PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)


# %%
# Define paths
PROJECT_PATH = Path("/home/whilebell/Code/football-tracker-analysis/")
SOURCE_VIDEO_PATH = PROJECT_PATH / "data/testing_video/08fd33_4.mp4"
OUTPUT_VIDEO_PATH = PROJECT_PATH / "data/output_videos/08fd33_4.mp4"

# %%
STRIDE = 30
PLAYER_ID = 2


def extract_crop(source_video_path: str):
    # Extracts crops of players from the video source using the player detection model.
    frame_generator = sv.get_video_frames_generator(source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc="Collecting crops"):
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
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
    for batch in tqdm(batches, desc="Extracting embeddings"):
        inputs = EMBEDDING_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
        outputs = EMBEDDING_MODEL(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        data.append(embeddings)

data = np.concatenate(data)
print(data.shape)


# %%
# frame generator for the video source
BALL_ID = 0
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
    thickness=2,
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
    text_color=sv.Color.from_hex("#000000"),
    text_position=sv.Position.BOTTOM_CENTER,
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#FFD700"), base=20, height=20
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
        all_other_detections.class_id = all_other_detections.class_id - 1
        all_other_detections = tracker.update_with_detections(all_other_detections)

        labels = [f"{tracker_id}" for tracker_id in all_other_detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(
            annotated_frame, all_other_detections
        )
        annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame, all_other_detections, labels=labels
        )

        video_sink.write_frame(annotated_frame)
