from uvicorn import Server, Config
from fastapi import FastAPI
# from starlette.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

import os

from yolofastapi.routers import yolo

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(yolo.router)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    server = Server(Config(app, host="127.0.0.1", port=port, lifespan="on"))
    server.run()