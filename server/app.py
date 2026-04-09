import uvicorn
import sys
import os

# This allows the script to find app.py in the root folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app 

def main():
    """The entry point the grader is looking for."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
