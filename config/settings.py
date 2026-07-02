import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    API_KEY: str = os.getenv("API_KEY", "")
    UNLOCK_KEY: str = os.getenv("UNLOCK_KEY", "")


settings = Settings()
