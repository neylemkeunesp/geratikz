from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    print("Starting server...")
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=3333,
            log_level="debug",
            access_log=True
        )
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise
