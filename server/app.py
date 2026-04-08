from app import app
import uvicorn
from app import app

def main():
    """The entry point that the grader is looking for."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
