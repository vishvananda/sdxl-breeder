from typing import Union, List

from fastapi import FastAPI, Body
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import engine
import os
import cv2
import shutil
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def read_root():
    return engine.recent()


class DeleteRequest(BaseModel):
    uuid: str

class MixRequest(BaseModel):
    uuid: List[str]
    n: int = 1

class PromptRequest(BaseModel):
    p: Union[str, None]
    n: int = 1

@app.post("/new")
async def new_prompt(request_body: PromptRequest):
    p = request_body.p
    n = request_body.n

    t = engine.gen_tok_embeds(p)
    uuid = engine.get_uuid()
    images = list(engine.gen(t, uuid, n))
    return {"p": p, "uuid": uuid, "images": images}

@app.post("/delete")
async def delete_uuid(request_body: DeleteRequest):
    uuid = request_body.uuid

    engine.delete_uuid(uuid)
    return {"uuid": uuid}


@app.get("/data/{image}")
async def get_image(image: str):
    return FileResponse(f"data/{image}")



@app.post("/mix")
async def mix_prompts(request_body: MixRequest):
    uuid = request_body.uuid
    n = request_body.n

    ts = [engine.load(u) for u in uuid]

    # Directory for temporary image storage
    temp_dir = f"data/temp_{engine.get_uuid()}"

    os.makedirs(temp_dir, exist_ok=True)
    
    max_image = 1001
    # Generate images for each mix ratio
    for i, ratio in enumerate(np.linspace(0, 1, max_image)):
        mixed_t = engine.mix_lerp(ts[0], ts[1], seed=42, ratio=ratio)
        image_filename = next(engine.gen(mixed_t, f"temp_{i}", n))
        os.rename(image_filename, os.path.join(temp_dir, f"{i:04d}.jpg"))

    # Combine images into a video
    video_filename = f"data/{engine.get_uuid()}.mp4"
    img_array = []
    for i in range(max_image):
        img_path = os.path.join(temp_dir, f"{i:04d}.jpg")
        img = cv2.imread(img_path)
        if i == 0:
            height, width, layers = img.shape
            size = (width, height)
        img_array.append(img)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 60, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    # Cleanup the temporary directory
    # Be careful with `shutil.rmtree`, ensure it's the correct directory to prevent accidental data loss
    shutil.rmtree(temp_dir)

    t = engine.mix_lerp(ts[0], ts[1], seed=42, ratio=0.5)
    uuid = engine.get_uuid()
    images = list(engine.gen(t, uuid, n))

    return {"uuid": uuid, "images": images}
