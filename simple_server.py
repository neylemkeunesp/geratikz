from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run("simple_server:app", host="localhost", port=3333, reload=True)
