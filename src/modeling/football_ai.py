# %%
import os
import supervision as sv
import torch
import numpy as np
from sports.common.team import TeamClassifier
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from transformers import AutoProcessor, SiglipVisionModel
from tqdm import tqdm
from pathlib import Path
from inference import get_model
from dotenv import load_dotenv
import cv2
from typing import Optional

load_dotenv()


# %%
# Load Player Detection Model from Roboflow
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/12"
PLAYER_DETECTION_MODEL = get_model(PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

# %%
# Load Pich Detection Model from Roboflow
FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/15"
FIELD_DETECTION_MODEL = get_model(
    model_id=FIELD_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY
)


# %%
# Load Models from Hugging Face
MODEL_ID = "google/siglip-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(MODEL_ID).to(DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)


# %%
# Define paths
PROJECT_PATH = Path("/home/whilebell/Code/football-tracker-analysis/")
SOURCE_VIDEO_PATH = PROJECT_PATH / "data/testing_videos/121364_0.mp4"
OUTPUT_VIDEO_PATH = PROJECT_PATH / "data/output_videos/121364_0_MULTI_VIEW.mp4"

# %%
STRIDE = 1  # Process every frame for smooth video
PLAYER_ID = 2


def extract_crop(source_video_path: str):
    # Extracts crops of players from the video source using the player detection model.
    frame_generator = sv.get_video_frames_generator(
        source_video_path, stride=30
    )  # Use larger stride for crop extraction

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
def resolve_goalkeepers_team_id(
    players_detections: sv.Detections, goalkeeper_detections: sv.Detections
):
    if len(goalkeeper_detections) == 0:
        return np.array([], dtype=int)

    goalkeepers_xy = goalkeeper_detections.get_anchors_coordinates(
        sv.Position.BOTTOM_CENTER
    )
    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    if (
        len(players_xy[players_detections.class_id == 0]) == 0
        or len(players_xy[players_detections.class_id == 1]) == 0
    ):
        return np.zeros(len(goalkeeper_detections), dtype=int)

    team_0_centroid = players_xy[players_detections.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_detections.class_id == 1].mean(axis=0)

    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id, dtype=int)


# %%
def draw_pitch_voronoi_diagram_2(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draws a Voronoi diagram on a soccer pitch representing the control areas of two
    teams with smooth color transitions.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    if len(team_1_xy) == 0 or len(team_2_xy) == 0:
        return pitch

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices(
        (scaled_width + 2 * padding, scaled_length + 2 * padding)
    )

    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt(
            (xy[:, 0][:, None, None] * scale - x_coordinates) ** 2
            + (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2
        )

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    # Increase steepness of the blend effect
    steepness = 15  # Increased steepness for sharper transition
    distance_ratio = min_distances_team_2 / np.clip(
        min_distances_team_1 + min_distances_team_2, a_min=1e-5, a_max=None
    )
    blend_factor = np.tanh((distance_ratio - 0.5) * steepness) * 0.5 + 0.5

    # Create the smooth color transition
    for c in range(3):  # Iterate over the B, G, R channels
        voronoi[:, :, c] = (
            blend_factor * team_1_color_bgr[c]
            + (1 - blend_factor) * team_2_color_bgr[c]
        ).astype(np.uint8)

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay


def resize_frame(frame, target_width, target_height):
    """Resize frame to target dimensions while maintaining aspect ratio"""
    h, w = frame.shape[:2]

    # Calculate scaling factor to fit within target dimensions
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h))

    # Create canvas with target dimensions
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Center the resized frame on the canvas
    start_y = (target_height - new_h) // 2
    start_x = (target_width - new_w) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized

    return canvas


def create_multi_view_frame(original_frame, topdown_frame, voronoi_frame, video_info):
    """Create a multi-view frame according to the layout"""
    video_height, video_width = video_info.height, video_info.width

    # Define the layout dimensions
    # Top row: original frame (left half) + topdown (right half)
    # Bottom row: voronoi (left half) + blank (right half)

    half_width = video_width // 2
    half_height = video_height // 2

    # Resize frames to fit their designated areas
    original_resized = resize_frame(original_frame, half_width, half_height)
    topdown_resized = resize_frame(topdown_frame, half_width, half_height)
    voronoi_resized = resize_frame(voronoi_frame, half_width, half_height)

    # Create blank area (black)
    blank_area = np.zeros((half_height, half_width, 3), dtype=np.uint8)

    # Combine frames according to layout
    # Top row
    top_row = np.hstack([original_resized, topdown_resized])
    # Bottom row
    bottom_row = np.hstack([voronoi_resized, blank_area])

    # Combine top and bottom rows
    multi_view_frame = np.vstack([top_row, bottom_row])

    return multi_view_frame


def process_frame(frame, team_classifier, tracker, CONFIG):
    """Process a single frame and return all three views"""

    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3

    # Detection and tracking
    result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)

    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections[detections.class_id != BALL_ID]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections = tracker.update_with_detections(detections=all_detections)

    goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
    players_detections = all_detections[all_detections.class_id == PLAYER_ID]
    referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

    # Team assignment
    if len(players_detections) > 0:
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops).astype(int)

        goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
            players_detections, goalkeepers_detections
        ).astype(int)

    referees_detections.class_id = np.full(
        len(referees_detections), 2, dtype=int
    )  # Set referees to class 2

    all_detections = sv.Detections.merge(
        [players_detections, goalkeepers_detections, referees_detections]
    )

    # Ensure all class_ids are integers
    if len(all_detections) > 0:
        all_detections.class_id = all_detections.class_id.astype(int)

    # Annotators
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]), thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
        text_color=sv.Color.from_hex("#000000"),
        text_position=sv.Position.BOTTOM_CENTER,
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"), base=20, height=17
    )

    # Create annotated original frame
    labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]
    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame, detections=all_detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=all_detections, labels=labels
    )
    annotated_frame = triangle_annotator.annotate(
        scene=annotated_frame, detections=ball_detections
    )

    # Detect pitch key points and transform coordinates
    try:
        result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result)

        if len(key_points.xy) > 0 and len(key_points.xy[0]) > 0:
            filter = key_points.confidence[0] > 0.5
            if np.any(filter):
                frame_reference_points = key_points.xy[0][filter]
                pitch_reference_points = np.array(CONFIG.vertices)[filter]

                if (
                    len(frame_reference_points) >= 4
                ):  # Need at least 4 points for transformation
                    transformer = ViewTransformer(
                        source=frame_reference_points, target=pitch_reference_points
                    )

                    # Transform coordinates
                    players_detections_all = sv.Detections.merge(
                        [players_detections, goalkeepers_detections]
                    )

                    # Ensure class_ids are integers for merged detections
                    if len(players_detections_all) > 0:
                        players_detections_all.class_id = (
                            players_detections_all.class_id.astype(int)
                        )

                    pitch_ball_xy = np.array([]).reshape(0, 2)
                    if len(ball_detections) > 0:
                        frame_ball_xy = ball_detections.get_anchors_coordinates(
                            sv.Position.BOTTOM_CENTER
                        )
                        pitch_ball_xy = transformer.transform_points(
                            points=frame_ball_xy
                        )

                    pitch_players_xy = np.array([]).reshape(0, 2)
                    if len(players_detections_all) > 0:
                        players_xy = players_detections_all.get_anchors_coordinates(
                            sv.Position.BOTTOM_CENTER
                        )
                        pitch_players_xy = transformer.transform_points(
                            points=players_xy
                        )

                    pitch_referees_xy = np.array([]).reshape(0, 2)
                    if len(referees_detections) > 0:
                        referees_xy = referees_detections.get_anchors_coordinates(
                            sv.Position.BOTTOM_CENTER
                        )
                        pitch_referees_xy = transformer.transform_points(
                            points=referees_xy
                        )

                    # Create top-down view
                    topdown_frame = draw_pitch(CONFIG)
                    if len(pitch_ball_xy) > 0:
                        topdown_frame = draw_points_on_pitch(
                            config=CONFIG,
                            xy=pitch_ball_xy,
                            face_color=sv.Color.WHITE,
                            edge_color=sv.Color.BLACK,
                            radius=10,
                            pitch=topdown_frame,
                        )
                    if len(pitch_players_xy) > 0:
                        team_0_mask = players_detections_all.class_id == 0
                        team_1_mask = players_detections_all.class_id == 1

                        if np.any(team_0_mask):
                            topdown_frame = draw_points_on_pitch(
                                config=CONFIG,
                                xy=pitch_players_xy[team_0_mask],
                                face_color=sv.Color.from_hex("00BFFF"),
                                edge_color=sv.Color.BLACK,
                                radius=16,
                                pitch=topdown_frame,
                            )
                        if np.any(team_1_mask):
                            topdown_frame = draw_points_on_pitch(
                                config=CONFIG,
                                xy=pitch_players_xy[team_1_mask],
                                face_color=sv.Color.from_hex("FF1493"),
                                edge_color=sv.Color.BLACK,
                                radius=16,
                                pitch=topdown_frame,
                            )
                    if len(pitch_referees_xy) > 0:
                        topdown_frame = draw_points_on_pitch(
                            config=CONFIG,
                            xy=pitch_referees_xy,
                            face_color=sv.Color.from_hex("FFD700"),
                            edge_color=sv.Color.BLACK,
                            radius=16,
                            pitch=topdown_frame,
                        )

                    # Create Voronoi diagram
                    voronoi_frame = draw_pitch(
                        config=CONFIG,
                        background_color=sv.Color.WHITE,
                        line_color=sv.Color.BLACK,
                    )

                    if len(pitch_players_xy) > 0:
                        team_0_mask = players_detections_all.class_id == 0
                        team_1_mask = players_detections_all.class_id == 1

                        team_0_xy = (
                            pitch_players_xy[team_0_mask]
                            if np.any(team_0_mask)
                            else np.array([]).reshape(0, 2)
                        )
                        team_1_xy = (
                            pitch_players_xy[team_1_mask]
                            if np.any(team_1_mask)
                            else np.array([]).reshape(0, 2)
                        )

                        voronoi_frame = draw_pitch_voronoi_diagram_2(
                            config=CONFIG,
                            team_1_xy=team_0_xy,
                            team_2_xy=team_1_xy,
                            team_1_color=sv.Color.from_hex("00BFFF"),
                            team_2_color=sv.Color.from_hex("FF1493"),
                            pitch=voronoi_frame,
                        )

                        # Add points on top of Voronoi
                        if len(pitch_ball_xy) > 0:
                            voronoi_frame = draw_points_on_pitch(
                                config=CONFIG,
                                xy=pitch_ball_xy,
                                face_color=sv.Color.WHITE,
                                edge_color=sv.Color.WHITE,
                                radius=8,
                                thickness=1,
                                pitch=voronoi_frame,
                            )
                        if len(team_0_xy) > 0:
                            voronoi_frame = draw_points_on_pitch(
                                config=CONFIG,
                                xy=team_0_xy,
                                face_color=sv.Color.from_hex("00BFFF"),
                                edge_color=sv.Color.WHITE,
                                radius=16,
                                thickness=1,
                                pitch=voronoi_frame,
                            )
                        if len(team_1_xy) > 0:
                            voronoi_frame = draw_points_on_pitch(
                                config=CONFIG,
                                xy=team_1_xy,
                                face_color=sv.Color.from_hex("FF1493"),
                                edge_color=sv.Color.WHITE,
                                radius=16,
                                thickness=1,
                                pitch=voronoi_frame,
                            )

                    return annotated_frame, topdown_frame, voronoi_frame
    except Exception as e:
        print(f"Error in pitch detection/transformation: {e}")

    # Fallback: return original frame and empty pitch views
    empty_pitch = draw_pitch(CONFIG)
    return annotated_frame, empty_pitch, empty_pitch


# %%
# Main execution
print("Extracting crops for team classification...")
crops = extract_crop(SOURCE_VIDEO_PATH)
team_classifier = TeamClassifier(device=DEVICE)
team_classifier.fit(crops)

CONFIG = SoccerPitchConfiguration()

# Initialize tracker
tracker = sv.ByteTrack()

# Get video info
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(
    f"Video info: {video_info.width}x{video_info.height}, {video_info.fps} fps, {video_info.total_frames} frames"
)

# Create video writer
with sv.VideoSink(str(OUTPUT_VIDEO_PATH), video_info) as sink:
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=STRIDE)

    for frame in tqdm(
        frame_generator, desc="Processing video", total=video_info.total_frames
    ):
        # Process frame to get all three views
        original_view, topdown_view, voronoi_view = process_frame(
            frame, team_classifier, tracker, CONFIG
        )

        # Create multi-view frame
        multi_view_frame = create_multi_view_frame(
            original_view, topdown_view, voronoi_view, video_info
        )

        # Write frame to output video
        sink.write_frame(multi_view_frame)

print(f"Multi-view video saved to: {OUTPUT_VIDEO_PATH}")
