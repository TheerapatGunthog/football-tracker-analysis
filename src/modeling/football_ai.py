# %%
import os
import supervision as sv
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
PROJECT_PATH = Path("/home/whilebell/Code/football-tracker-analysis/")
SOURCE_VIDEO_PATH = PROJECT_PATH / "data/testing_video/08fd33_4.mp4"
OUTPUT_VIDEO_PATH = PROJECT_PATH / "data/output_videos/08fd33_4.mp4"


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
