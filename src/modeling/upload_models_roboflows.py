import roboflow
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration (IMPORTANT: UPDATE THESE VALUES BEFORE RUNNING) ---
# 1. Get your API Key from your Roboflow account settings.
API_KEY = os.getenv("ROBOFLOW_API_KEY")

# 2. Get your Workspace ID and Project ID from your Roboflow project URL.
#    Example URL: https://app.roboflow.com/YOUR_WORKSPACE_ID/YOUR_PROJECT_ID/
WORKSPACE_ID = "footballai-2cnp1"
PROJECT_ID = "football-players-detection-3zvbc-nzxsd"

# 3. Choose the dataset version number you want to associate this model with.
#    This version should ideally NOT have a model already uploaded or trained.
DATASET_VERSION_NUMBER = 1  # e.g., 1, 2, 3

# 4. Define the type of model you are uploading.
#    Given your file structure with 'best.pt' and 'train/weights', it's likely YOLOv8.
#    Confirm this matches the actual architecture of your trained model.
MODEL_TYPE = "yolov8"  # Common examples: "yolov8", "yolov5", "rfdetr-base"

# 5. Define the path to the directory containing your model weights.
#    Based on the image you provided, the path to the 'weights' directory.
MODEL_DIRECTORY_PATH = "/home/whilebell/Code/football-tracker-analysis/models/soccer_player_detector/train/weights/"

# 6. Define the name of your weights file.
#    Based on the image you provided, it's 'best.pt'.
WEIGHTS_FILENAME = "best.pt"


# --- Function to upload the model ---
def upload_model_to_roboflow():
    """
    Uploads a locally trained model to a specified Roboflow project and dataset version.
    """
    try:
        print("--- Initiating Model Upload to Roboflow ---")
        print(
            f"Targeting Workspace: {WORKSPACE_ID}, Project: {PROJECT_ID}, Version: {DATASET_VERSION_NUMBER}"
        )
        print(
            f"Model Type: {MODEL_TYPE}, Model Directory: {MODEL_DIRECTORY_PATH}, Weights File: {WEIGHTS_FILENAME}"
        )

        # 1. Initialize Roboflow with your API key
        # Alternatively, you can run `roboflow.login()` in your terminal/console
        # before running this script, and it will handle authentication.
        rf = roboflow.Roboflow(api_key=API_KEY)
        print("Roboflow initialized successfully.")

        # 2. Get your specific workspace
        workspace = rf.workspace(WORKSPACE_ID)
        print(f"Accessed workspace: '{workspace.name}'")

        # 3. Get your specific project within the workspace
        project = workspace.project(PROJECT_ID)
        print(f"Accessed project: '{project.name}'")

        # 4. Select the dataset version to associate the model with
        # This version should ideally not have an existing model deployed to it.
        version = project.version(DATASET_VERSION_NUMBER)
        print(
            f"Selected dataset version: {version.version} (dataset name: {version.name})"
        )

        # 5. Verify the existence of the model weights file locally
        full_weights_file_path = os.path.join(MODEL_DIRECTORY_PATH, WEIGHTS_FILENAME)
        if not os.path.exists(full_weights_file_path):
            print(
                f"\nERROR: Model weights file not found at '{full_weights_file_path}'"
            )
            print(
                "Please ensure 'MODEL_DIRECTORY_PATH' and 'WEIGHTS_FILENAME' are correct."
            )
            print(
                "The script needs to be run from a directory where 'models/soccer_player_detector/train/weights/best.pt' is a valid path."
            )
            return

        print(f"Found model weights file: {full_weights_file_path}")

        # 6. Deploy your model to Roboflow
        # This function uploads the weights to Roboflow's servers.
        print("\nUploading model weights to Roboflow. This may take a few moments...")
        version.deploy(
            model_type=MODEL_TYPE,
            model_path=MODEL_DIRECTORY_PATH,
            filename=WEIGHTS_FILENAME,
        )

        print("\n--- Model Upload Initiated Successfully! ---")
        print(
            f"Your model for dataset version {DATASET_VERSION_NUMBER} of project '{PROJECT_ID}' is now being processed by Roboflow."
        )
        print(
            "You can monitor its status and prepare for deployment on your Roboflow dashboard:"
        )
        print(
            f"Visit: [https://app.roboflow.com/](https://app.roboflow.com/){WORKSPACE_ID}/{PROJECT_ID}/models"
        )
        print(
            "\nOnce processed, you can use Roboflow's inference API or SDK to make predictions with your model."
        )

    except Exception as e:
        print("\n--- An Error Occurred During Model Upload ---")
        print(f"Error Details: {e}")
        print("Please check the following:")
        print(
            "  - Your API_KEY, WORKSPACE_ID, PROJECT_ID, and DATASET_VERSION_NUMBER are correct."
        )
        print(
            "  - The MODEL_DIRECTORY_PATH and WEIGHTS_FILENAME are accurate and the file exists."
        )
        print(
            "  - The DATASET_VERSION_NUMBER does not already have a model deployed to it (Roboflow typically allows one model per version)."
        )
        print("  - Your internet connection is stable.")


# Run the upload function when the script is executed
if __name__ == "__main__":
    upload_model_to_roboflow()
