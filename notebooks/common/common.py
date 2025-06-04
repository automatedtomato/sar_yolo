from dotenv import load_dotenv
import os
import sys
from constants import PROJECT_ROOT


def setup_imports() -> str:
    """
    Setup the import path
    """

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path)

    if project_root not in sys.path:
        sys.path.append(project_root)

    return project_root
