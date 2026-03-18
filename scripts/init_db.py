#!/usr/bin/env python3
"""Initialize the execution database."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from store.db import init_db
from config.settings import settings

if __name__ == "__main__":
    print(f"Initializing DB at {settings.db_path}")
    init_db()
    print("Database initialized successfully")
