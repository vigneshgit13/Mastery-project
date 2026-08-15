from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / ".env")


class Settings:
    PROJECT_ID: str = os.getenv("PROJECT_ID", "")
    BUCKET_NAME: str = os.getenv("BUCKET_NAME", "")
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS", ""
    )

    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "25"))

    ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png"
}




settings = Settings()