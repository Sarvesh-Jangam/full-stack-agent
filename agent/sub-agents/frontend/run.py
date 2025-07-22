# run.py
import uvicorn
from main import root_agent

if __name__ == "__main__":
    print("Attempting to start the web server programmatically...")
    try:
        # The ADK web server is a FastAPI app, which can be run with uvicorn
        uvicorn.run(
            "google.adk.runtime.app:create_app",
            factory=True,
            host="127.0.0.1",
            port=8000,
            reload=True,
        )
        print("Server should be running at http://127.0.0.1:8000")
    except Exception as e:
        print(f"An error occurred: {e}")