from typing import Union, List

from fastapi import FastAPI, Body
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import engine

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
    t = engine.mix(*ts)
    uuid = engine.get_uuid()
    images = list(engine.gen(t, uuid, n))
    return {"uuid": uuid, "images": images}
