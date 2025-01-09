import sys
import traceback

try:
    from fastapi import FastAPI
    import uvicorn
    print("FastAPI and uvicorn imported successfully")
    
    app = FastAPI()
    print("FastAPI app created")

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    if __name__ == "__main__":
        print("Starting server...")
        sys.stdout.flush()
        uvicorn.run(app, host="127.0.0.1", port=3333, log_level="debug")
except Exception as e:
    print("Error during startup:", str(e))
    traceback.print_exc()
    sys.exit(1)
