import os
from dotenv import load_dotenv
from SoccerNet.Downloader import SoccerNetDownloader

load_dotenv()

soccernet_password = os.getenv("SoccerNet_Password")
print(f"Using SoccerNet password: {soccernet_password}")

mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="/home/whilebell/Code/football-tracker-analysis/data/soccernet-tracking/"
)

mySoccerNetDownloader.password = "soccernet_password"
mySoccerNetDownloader.downloadDataTask(
    task="tracking", split=["train", "test", "challenge"]
)
mySoccerNetDownloader.downloadDataTask(
    task="tracking-2023", split=["train", "test", "challenge"]
)
