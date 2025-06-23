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
from typing import Optional, Dict, Any

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
PROJECT_PATH = Path(
    "/home/whilebell/Code/football-tracker-analysis/"
)  # Adjusted for local execution
SOURCE_VIDEO_PATH = PROJECT_PATH / "data/testing_videos/121364_0.mp4"
OUTPUT_VIDEO_PATH = PROJECT_PATH / "data/output_videos/121364_0_MULTI_VIEW_STATS_50.mp4"

# Create output directory if it doesn't exist
OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)


# %%
# --- NEW --- Faster Testing Flag
TEST_MODE = False  # Set to True to process only the first 10 seconds of the video

STRIDE = 1  # Process every frame for smooth video
PLAYER_ID = 2


def extract_crop(source_video_path: str):
    """Extracts crops of players from the video source using the player detection model."""
    frame_generator = sv.get_video_frames_generator(
        source_video_path, stride=30
    )  # Use larger stride for crop extraction

    crops = []
    for frame in frame_generator:
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
    """Assigns goalkeepers to the nearest team based on centroid distance."""
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

    steepness = 15
    distance_ratio = min_distances_team_2 / np.clip(
        min_distances_team_1 + min_distances_team_2, a_min=1e-5, a_max=None
    )
    blend_factor = np.tanh((distance_ratio - 0.5) * steepness) * 0.5 + 0.5

    for c in range(3):
        voronoi[:, :, c] = (
            blend_factor * team_1_color_bgr[c]
            + (1 - blend_factor) * team_2_color_bgr[c]
        ).astype(np.uint8)

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay


# Updated draw_stats_panel function with debug info
def draw_stats_panel(
    stats: Dict,
    width: int,
    height: int,
    team_0_color: tuple,
    team_1_color: tuple,
) -> np.ndarray:
    """
    Draws a panel displaying team statistics with optional debug information.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_color = (255, 255, 255)  # White

    y_offset = 40
    line_height = 35

    # Team 0 Stats (Blue)
    cv2.putText(
        panel, "Team Blue", (30, y_offset), font, font_scale, team_0_color, thickness
    )
    y_offset += line_height
    cv2.putText(
        panel,
        f"Possession: {stats['possession'][0]:.1f}%",
        (30, y_offset),
        font,
        font_scale * 0.8,
        text_color,
        thickness - 1,
    )
    y_offset += line_height - 5
    cv2.putText(
        panel,
        f"Distance: {stats['distance'][0]:.1f} m",
        (30, y_offset),
        font,
        font_scale * 0.8,
        text_color,
        thickness - 1,
    )
    y_offset += line_height - 5
    cv2.putText(
        panel,
        f"Avg Speed: {stats['avg_speed'][0]:.2f} m/s",
        (30, y_offset),
        font,
        font_scale * 0.8,
        text_color,
        thickness - 1,
    )
    y_offset += line_height - 5
    cv2.putText(
        panel,
        f"Passes: {stats['passes'][0]}",
        (30, y_offset),
        font,
        font_scale * 0.8,
        text_color,
        thickness - 1,
    )

    # Team 1 Stats (Pink)
    y_offset = 40
    cv2.putText(
        panel,
        "Team Pink",
        (width // 2 + 30, y_offset),
        font,
        font_scale,
        team_1_color,
        thickness,
    )
    y_offset += line_height
    cv2.putText(
        panel,
        f"Possession: {stats['possession'][1]:.1f}%",
        (width // 2 + 30, y_offset),
        font,
        font_scale * 0.8,
        text_color,
        thickness - 1,
    )
    y_offset += line_height - 5
    cv2.putText(
        panel,
        f"Distance: {stats['distance'][1]:.1f} m",
        (width // 2 + 30, y_offset),
        font,
        font_scale * 0.8,
        text_color,
        thickness - 1,
    )
    y_offset += line_height - 5
    cv2.putText(
        panel,
        f"Avg Speed: {stats['avg_speed'][1]:.2f} m/s",
        (width // 2 + 30, y_offset),
        font,
        font_scale * 0.8,
        text_color,
        thickness - 1,
    )
    y_offset += line_height - 5
    cv2.putText(
        panel,
        f"Passes: {stats['passes'][1]}",
        (width // 2 + 30, y_offset),
        font,
        font_scale * 0.8,
        text_color,
        thickness - 1,
    )

    return panel


class TeamStatsTracker:
    """
    A class to track and calculate statistics for two teams in a soccer match.
    This version includes a definitive fix for the average speed calculation.
    """

    def __init__(self, video_info: sv.VideoInfo):
        """
        Initializes the tracker.
        """
        # --- General Stats ---
        self.team_distances = {0: 0.0, 1: 0.0}
        self.player_last_positions = {}  # {tracker_id: np.array([x, y])}
        self.time_per_frame = STRIDE / video_info.fps
        self.total_processed_time = 0.0

        self.player_frame_count = {0: 0, 1: 0}

        # --- Speed Calculation (only used for filtering, not for averaging) ---
        self.speed_sum = {0: 0.0, 1: 0.0}
        self.speed_count = {0: 0, 1: 0}

        # --- Enhanced Possession and Pass Tracking State ---
        self.possession_threshold_m = 300.0
        self.min_pass_distance_m = 1.5
        self.confirmation_frames_threshold = 3

        self.ball_state = "LOOSE"
        self.player_in_control_id = None
        self.team_in_control = None
        self.last_player_in_control_id = None

        self.potential_player_id = None
        self.possession_confirmation_counter = 0

        self.team_possession_time = {0: 0.0, 1: 0.0}
        self.team_passes = {0: 0, 1: 0}

    def _get_player_team_map(self, detections: sv.Detections) -> Dict[int, int]:
        if detections.tracker_id is None:
            return {}
        return {
            tracker_id: class_id
            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)
        }

    def _update_distances_and_speeds(
        self,
        detections: sv.Detections,
        pitch_players_xy: np.ndarray,
        player_team_map: Dict[int, int],
    ):
        if detections.tracker_id is None or len(pitch_players_xy) == 0:
            return

        for i, tracker_id in enumerate(detections.tracker_id):
            if i >= len(pitch_players_xy):
                continue

            team_id = player_team_map.get(tracker_id)
            if team_id is None or team_id not in self.team_distances:
                continue

            self.player_frame_count[team_id] += 1

            current_pos = pitch_players_xy[i]
            if tracker_id in self.player_last_positions:
                dist_moved = np.linalg.norm(
                    current_pos - self.player_last_positions[tracker_id]
                )
                if (
                    0.1 < dist_moved < 5.0
                ):  # This threshold is fine for distance accumulation
                    self.team_distances[team_id] += dist_moved

                    # We keep this part to ensure the speed_count variable still exists,
                    # but we will NOT use it for the final average speed calculation.
                    speed = dist_moved / self.time_per_frame
                    self.speed_sum[team_id] += speed
                    self.speed_count[team_id] += 1

            self.player_last_positions[tracker_id] = current_pos

    # (The _update_possession method remains the same as the one I provided in the previous step)
    def _update_possession(
        self,
        detections: sv.Detections,
        pitch_players_xy: np.ndarray,
        pitch_ball_xy: np.ndarray,
        player_team_map: Dict[int, int],
    ):
        if (
            len(pitch_ball_xy) == 0
            or len(pitch_players_xy) == 0
            or detections.tracker_id is None
        ):
            if self.ball_state == "CONTROLLED":
                self.last_player_in_control_id = self.player_in_control_id
            self.ball_state = "LOOSE"
            self.player_in_control_id = None
            self.team_in_control = None
            self.possession_confirmation_counter = 0
            return

        ball_pos = pitch_ball_xy[0]
        distances_to_ball = np.linalg.norm(pitch_players_xy - ball_pos, axis=1)
        player_idx_closest = np.argmin(distances_to_ball)
        min_dist_to_ball = distances_to_ball[player_idx_closest]

        if min_dist_to_ball < self.possession_threshold_m:
            current_closest_player_id = detections.tracker_id[player_idx_closest]
            if self.potential_player_id == current_closest_player_id:
                self.possession_confirmation_counter += 1
            else:
                self.potential_player_id = current_closest_player_id
                self.possession_confirmation_counter = 1

            if (
                self.possession_confirmation_counter
                >= self.confirmation_frames_threshold
            ):
                new_player_in_control_id = self.potential_player_id
                new_team_in_control = player_team_map.get(new_player_in_control_id)
                if new_team_in_control is not None and new_team_in_control in [0, 1]:
                    if (
                        self.ball_state == "LOOSE"
                        and self.last_player_in_control_id is not None
                        and self.last_player_in_control_id != new_player_in_control_id
                    ):
                        last_team = player_team_map.get(self.last_player_in_control_id)
                        if last_team == new_team_in_control:
                            last_pos = self.player_last_positions.get(
                                self.last_player_in_control_id, None
                            )
                            current_pos = self.player_last_positions.get(
                                new_player_in_control_id, None
                            )
                            if last_pos is not None and current_pos is not None:
                                pass_dist = np.linalg.norm(last_pos - current_pos)
                                if pass_dist > self.min_pass_distance_m:
                                    self.team_passes[new_team_in_control] += 1
                    self.ball_state = "CONTROLLED"
                    self.player_in_control_id = new_player_in_control_id
                    self.team_in_control = new_team_in_control
        else:
            if self.ball_state == "CONTROLLED":
                self.last_player_in_control_id = self.player_in_control_id
            self.ball_state = "LOOSE"
            self.player_in_control_id = None
            self.team_in_control = None
            self.possession_confirmation_counter = 0

        if self.ball_state == "CONTROLLED" and self.team_in_control is not None:
            self.team_possession_time[self.team_in_control] += self.time_per_frame

    # (The update method remains the same)
    def update(
        self,
        detections: sv.Detections,
        pitch_players_xy: np.ndarray,
        pitch_ball_xy: np.ndarray,
    ):
        self.total_processed_time += self.time_per_frame
        if len(detections) == 0:
            if self.ball_state == "CONTROLLED":
                self.last_player_in_control_id = self.player_in_control_id
            self.ball_state = "LOOSE"
            self.player_in_control_id = None
            self.team_in_control = None
            return
        player_team_map = self._get_player_team_map(detections)
        self._update_distances_and_speeds(detections, pitch_players_xy, player_team_map)
        self._update_possession(
            detections, pitch_players_xy, pitch_ball_xy, player_team_map
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves the current statistics with a definitively corrected average speed calculation.
        """
        # --- Correct Average Speed Calculation ---
        # Total Distance / (Total Player-Frames * Time Per Frame)
        total_player_time_0 = self.player_frame_count[0] * self.time_per_frame
        total_player_time_1 = self.player_frame_count[1] * self.time_per_frame

        avg_speed_0 = (
            self.team_distances[0] / total_player_time_0
            if total_player_time_0 > 0
            else 0
        )
        avg_speed_1 = (
            self.team_distances[1] / total_player_time_1
            if total_player_time_1 > 0
            else 0
        )

        # --- Possession Calculation ---
        total_possession_time = (
            self.team_possession_time[0] + self.team_possession_time[1]
        )
        if total_possession_time > 0:
            possession_0 = (self.team_possession_time[0] / total_possession_time) * 100
            possession_1 = (self.team_possession_time[1] / total_possession_time) * 100
        else:
            possession_0 = possession_1 = 0.0

        return {
            "possession": {0: possession_0, 1: possession_1},
            "distance": self.team_distances,
            "avg_speed": {0: avg_speed_0, 1: avg_speed_1},
            "passes": self.team_passes,
            "debug": {
                "total_time": self.total_processed_time,
                "ball_state": self.ball_state,
                "player_in_control": self.player_in_control_id,
                "team_in_control": self.team_in_control,
                "possession_time_0": self.team_possession_time[0],
                "possession_time_1": self.team_possession_time[1],
            },
        }


def resize_frame(frame, target_width, target_height):
    """Resize frame to target dimensions while maintaining aspect ratio"""
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    start_y = (target_height - new_h) // 2
    start_x = (target_width - new_w) // 2
    canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized
    return canvas


def create_multi_view_frame(
    original_frame, topdown_frame, voronoi_frame, stats_panel, video_info
):
    """Create a multi-view frame according to the layout"""
    video_height, video_width = video_info.height, video_info.width
    half_width = video_width // 2
    half_height = video_height // 2

    original_resized = resize_frame(original_frame, half_width, half_height)
    topdown_resized = resize_frame(topdown_frame, half_width, half_height)
    voronoi_resized = resize_frame(voronoi_frame, half_width, half_height)
    stats_panel_resized = cv2.resize(stats_panel, (half_width, half_height))

    top_row = np.hstack([original_resized, topdown_resized])
    bottom_row = np.hstack([voronoi_resized, stats_panel_resized])
    multi_view_frame = np.vstack([top_row, bottom_row])

    return multi_view_frame


def process_frame(frame, team_classifier, tracker, CONFIG, stats_tracker):
    """Process a single frame and return all views and statistics"""

    BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID = 0, 1, 2, 3

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

    if len(players_detections) > 0:
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops).astype(int)
        goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
            players_detections, goalkeepers_detections
        ).astype(int)

    referees_detections.class_id = np.full(len(referees_detections), 2, dtype=int)

    all_detections = sv.Detections.merge(
        [players_detections, goalkeepers_detections, referees_detections]
    )
    if len(all_detections) > 0:
        all_detections.class_id = all_detections.class_id.astype(int)

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

    labels = (
        [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]
        if all_detections.tracker_id is not None
        else []
    )
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

    topdown_frame = draw_pitch(CONFIG)
    voronoi_frame = draw_pitch(
        CONFIG, background_color=sv.Color.WHITE, line_color=sv.Color.BLACK
    )
    current_stats = stats_tracker.get_stats()

    try:
        result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result)

        if len(key_points.xy) > 0 and len(key_points.xy[0]) > 0:
            filter_ = key_points.confidence[0] > 0.5
            if np.sum(filter_) >= 4:
                frame_reference_points = key_points.xy[0][filter_]
                pitch_reference_points = np.array(CONFIG.vertices)[filter_]
                transformer = ViewTransformer(
                    source=frame_reference_points, target=pitch_reference_points
                )

                players_detections_all = sv.Detections.merge(
                    [players_detections, goalkeepers_detections]
                )
                if len(players_detections_all) > 0:
                    players_detections_all.class_id = (
                        players_detections_all.class_id.astype(int)
                    )

                frame_ball_xy = ball_detections.get_anchors_coordinates(
                    sv.Position.BOTTOM_CENTER
                )
                pitch_ball_xy = (
                    transformer.transform_points(points=frame_ball_xy)
                    if len(frame_ball_xy) > 0
                    else np.array([]).reshape(0, 2)
                )

                players_xy = players_detections_all.get_anchors_coordinates(
                    sv.Position.BOTTOM_CENTER
                )
                pitch_players_xy = (
                    transformer.transform_points(points=players_xy)
                    if len(players_xy) > 0
                    else np.array([]).reshape(0, 2)
                )

                referees_xy = referees_detections.get_anchors_coordinates(
                    sv.Position.BOTTOM_CENTER
                )
                pitch_referees_xy = (
                    transformer.transform_points(points=referees_xy)
                    if len(referees_xy) > 0
                    else np.array([]).reshape(0, 2)
                )

                if len(players_detections_all) > 0 and len(pitch_players_xy) > 0:
                    stats_tracker.update(
                        players_detections_all, pitch_players_xy, pitch_ball_xy
                    )

                topdown_frame = draw_pitch(CONFIG)
                team_0_mask = players_detections_all.class_id == 0
                team_1_mask = players_detections_all.class_id == 1

                if pitch_ball_xy.shape[0] > 0:
                    topdown_frame = draw_points_on_pitch(
                        config=CONFIG,
                        xy=pitch_ball_xy,
                        face_color=sv.Color.WHITE,
                        edge_color=sv.Color.BLACK,
                        radius=10,
                        pitch=topdown_frame,
                    )
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
                if pitch_referees_xy.shape[0] > 0:
                    topdown_frame = draw_points_on_pitch(
                        config=CONFIG,
                        xy=pitch_referees_xy,
                        face_color=sv.Color.from_hex("FFD700"),
                        edge_color=sv.Color.BLACK,
                        radius=16,
                        pitch=topdown_frame,
                    )

                voronoi_frame = draw_pitch(
                    CONFIG, background_color=sv.Color.WHITE, line_color=sv.Color.BLACK
                )
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

                if pitch_ball_xy.shape[0] > 0:
                    voronoi_frame = draw_points_on_pitch(
                        config=CONFIG,
                        xy=pitch_ball_xy,
                        face_color=sv.Color.WHITE,
                        edge_color=sv.Color.WHITE,
                        radius=8,
                        thickness=1,
                        pitch=voronoi_frame,
                    )
                if team_0_xy.shape[0] > 0:
                    voronoi_frame = draw_points_on_pitch(
                        config=CONFIG,
                        xy=team_0_xy,
                        face_color=sv.Color.from_hex("00BFFF"),
                        edge_color=sv.Color.WHITE,
                        radius=16,
                        thickness=1,
                        pitch=voronoi_frame,
                    )
                if team_1_xy.shape[0] > 0:
                    voronoi_frame = draw_points_on_pitch(
                        config=CONFIG,
                        xy=team_1_xy,
                        face_color=sv.Color.from_hex("FF1493"),
                        edge_color=sv.Color.WHITE,
                        radius=16,
                        thickness=1,
                        pitch=voronoi_frame,
                    )

    except Exception as e:
        print(f"Error in pitch detection/transformation: {e}")

    current_stats = stats_tracker.get_stats()
    return annotated_frame, topdown_frame, voronoi_frame, current_stats


# %%
# Main execution
if not SOURCE_VIDEO_PATH.exists():
    print(f"Error: Source video not found at {SOURCE_VIDEO_PATH}")
else:
    print("Extracting crops for team classification...")
    crops = extract_crop(str(SOURCE_VIDEO_PATH))

    print("Fitting team classifier...")
    team_classifier = TeamClassifier(device=DEVICE)
    team_classifier.fit(crops)
    print("Team classifier fitted successfully.")

    CONFIG = SoccerPitchConfiguration()

    tracker = sv.ByteTrack()

    video_info = sv.VideoInfo.from_video_path(str(SOURCE_VIDEO_PATH))
    print(
        f"Video info: {video_info.width}x{video_info.height}, {video_info.fps} fps, {video_info.total_frames} frames"
    )

    stats_tracker = TeamStatsTracker(video_info)
    max_frames = int(10 * video_info.fps) if TEST_MODE else video_info.total_frames
    team_0_color = sv.Color.from_hex("00BFFF").as_bgr()
    team_1_color = sv.Color.from_hex("FF1493").as_bgr()

    with sv.VideoSink(str(OUTPUT_VIDEO_PATH), video_info) as sink:
        frame_generator = sv.get_video_frames_generator(
            str(SOURCE_VIDEO_PATH), stride=STRIDE
        )

        debug_interval = 100
        debug_detailed_interval = 500

        for i, frame in enumerate(
            tqdm(
                frame_generator,
                desc="Processing video",
                total=max_frames,
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        ):
            if i >= max_frames:
                break

            original_view, topdown_view, voronoi_view, stats = process_frame(
                frame, team_classifier, tracker, CONFIG, stats_tracker
            )

            stats_panel = draw_stats_panel(
                stats,
                video_info.width // 2,
                video_info.height // 2,
                team_0_color,
                team_1_color,
            )

            multi_view_frame = create_multi_view_frame(
                original_view, topdown_view, voronoi_view, stats_panel, video_info
            )

            sink.write_frame(multi_view_frame)

    print(f"Multi-view video with stats saved to: {OUTPUT_VIDEO_PATH}")
