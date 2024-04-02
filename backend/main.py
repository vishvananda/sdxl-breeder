from typing import Union, List

from fastapi import FastAPI, Body
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import engine
import cv2
import imagehash
import heapq
from collections import deque
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

    images = deque([load(u) for u in uuid])
    # Priority queue of (negative distance, index1, index2) to ensure max distance has highest priority
    pq = []

    # Initial population of the priority queue with distances between consecutive images
    for i in range(len(images) - 1):
        distance = calculate_distance(images[i], images[i + 1])
        heapq.heappush(pq, (-distance, images[i].uuid, images[i], images[i + 1]))
    
    num_images = 600

    while len(images) < num_images and pq:
        distance, _uuid, img1, img2 = heapq.heappop(pq)
        distance = -distance  # Convert back to positive since we stored it as negative

        print("Generating", distance) 
        new_image = generate_intermediate_image(img1, img2)

        # Find the new index for insertion
        new_index = images.index(img1) + 1

        bdistance = calculate_distance(images[new_index - 1], new_image)
        adistance = calculate_distance(new_image, images[new_index])

        # Don't add new image if we are not making progress
        if (bdistance == distance and adistance == 0.0) or (bdistance == 0.0 and adistance == distance):
            print("Skipping similar image at index", new_index)
            continue

        # Update priority queue with new distances
        print("Adding before", bdistance)
        heapq.heappush(pq, (-bdistance, images[new_index - 1].uuid, images[new_index - 1], new_image))
        print("Adding after", adistance)
        heapq.heappush(pq, (-adistance, new_image.uuid, new_image, images[new_index]))

        # Insert the new image into the deque
        print("Inserting at", new_index)
        images.insert(new_index, new_image)


    
    # Use the middle image from the list
    mid = images[len(images)// 2]
    video_filename = f"data/{mid.uuid}.mp4"

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 60, mid.img.size)

    for img in images:
        # Convert PIL Image to NumPy array and ensure it's in RGB format
        frame = np.array(img.img.convert('RGB'))
        # Convert RGB to BGR format for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

    out_images = list(engine.save_images(mid.t, mid.uuid, [mid.img]))
    return {"uuid": mid.uuid, "images": out_images}


class ImageData:
    __slots__ = ['uuid', 't', 'img']
    
    def __init__(self, uuid, t, img):
        self.uuid = uuid
        self.t = t
        self.img = img


def load(uuid):
    return ImageData(uuid, engine.load(uuid), engine.load_image(uuid))

import torch

def calculate_distance(image1, image2):
    """Calculate the distance between two images."""
    hash1 = imagehash.whash(image1.img)
    hash2 = imagehash.whash(image2.img)
    euc = torch.dist(image1.t[0], image2.t[0]).item() + torch.dist(image1.t[1], image2.t[1]).item()
    return hash1 - hash2 + euc


def generate_intermediate_image(image1, image2, seed=42):
    """Generate an intermediate image between two images."""
    mixed_t = engine.mix_slerp(image1.t, image2.t, seed=seed, ratio=0.5)
    image = engine.gen_images(mixed_t, 1, seed)[0]
    return ImageData(engine.get_uuid(), mixed_t, image)
