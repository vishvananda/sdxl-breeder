import os
from diffusers import AutoPipelineForText2Image
import torch
import glob
import uuid

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
DEVICE = "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"

pipeline = pipeline.to(DEVICE)


def get_tok_embed(prompt, tokenizer, text_encoder):
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
    ).to(DEVICE)
    return text_encoder.get_input_embeddings()(input_ids)


def gen_tok_embeds(prompt):
    lil = get_tok_embed(prompt, pipeline.tokenizer, pipeline.text_encoder)
    big = get_tok_embed(prompt, pipeline.tokenizer_2, pipeline.text_encoder_2)

    return lil, big


def encode_tokens(inputs_embeds, text_encoder):
    prompt_embeds = text_encoder(inputs_embeds=inputs_embeds, output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2].to("cpu")
    return prompt_embeds, pooled_prompt_embeds


def gen_prompt_embeds(tok_embeds):
    [lil_clip, big_clip] = tok_embeds

    pe1, _ = encode_tokens(lil_clip, pipeline.text_encoder)
    pe2, ppe2 = encode_tokens(big_clip, pipeline.text_encoder_2)

    prompt_embeds = torch.concat([pe1, pe2], dim=-1)
    pooled_prompt_embeds = ppe2  # SDXL only returns pooled prompt embeds text_encoder_2

    return prompt_embeds, pooled_prompt_embeds

def mix(t1, t2, seed=None, ratio=0.5):
    [l1, b1], [l2, b2] = t1, t2

    if seed is not None:
        torch.manual_seed(seed)

    # Calculate the exact split points, including partial index consideration
    l_exact_split = 768 * ratio
    b_exact_split = 1280 * ratio

    # Determine the integer split and the alpha for linear interpolation
    l_split_int, l_alpha = int(l_exact_split), l_exact_split % 1
    b_split_int, b_alpha = int(b_exact_split), b_exact_split % 1

    # Generate shuffled indices based on the seed
    l_indices = torch.randperm(768)[:l_split_int]
    b_indices = torch.randperm(1280)[:b_split_int]

    # Create copies of the original tensors to modify
    l_mixed = l1.clone()
    b_mixed = b1.clone()

    # Replace the fully selected elements
    l_mixed[:, :, l_indices] = l2[:, :, l_indices]
    b_mixed[:, :, b_indices] = b2[:, :, b_indices]

    # Linearly interpolate the partially selected index if alpha > 0
    if l_alpha > 0 and l_split_int < 768:
        l_mixed[:, :, l_split_int] = l1[:, :, l_split_int] * (1 - l_alpha) + l2[:, :, l_split_int] * l_alpha
    if b_alpha > 0 and b_split_int < 1280:
        b_mixed[:, :, b_split_int] = b1[:, :, b_split_int] * (1 - b_alpha) + b2[:, :, b_split_int] * b_alpha

    return [l_mixed, b_mixed]

def mix_lerp(t1, t2, seed=None, ratio=0.5):
    # Linear interpolation between two embeddings
    [l1, b1], [l2, b2] = t1, t2

    l = ratio * l2 + (1 - ratio) * l1
    b = ratio * b2 + (1 - ratio) * b1

    return [l, b]

def mix2(t1, t2, seed=None, mutations=100):
    # "mutate/crossover genetic-ish mixing"
    [l1, b1], [l2, b2] = t1, t2

    if seed is not None:
        torch.manual_seed(seed)

    if torch.rand(1) < 0.5:
        b1, b2 = b2, b1

    if torch.rand(1) < 0.5:
        l1, l2 = l2, l1

    l = torch.clone(l1).view(-1)
    b = torch.clone(b1).view(-1)

    idx = 0
    cur = False
    while idx < l.shape[0]:
        jump = torch.randint(500, 5000, size=(1,))
        maxl = min(l.shape[0] - idx, jump)
        crossover = torch.randint(0, maxl, size=(1,))
        if cur:
            l[idx : idx + crossover] = l2.view(-1)[idx : idx + crossover]
        cur = not cur
        idx += crossover + 1

    idx = 0
    cur = False
    while idx < b.shape[0]:
        jump = torch.randint(500, 5000, size=(1,))
        maxb = min(b.shape[0] - idx, jump)
        crossover = torch.randint(0, maxb, size=(1,))
        if cur:
            b[idx : idx + crossover] = b2.view(-1)[idx : idx + crossover]
        cur = not cur
        idx += crossover + 1

    return [l.view(l1.shape), b.view(b1.shape)]

    # for i in range(l.shape[0]):
    #     crossover = torch.randint(0, l.shape[0], size=(1,))
    #     l[i, crossover:] = l2[i, crossover:]

    # for i in range(b.shape[0]):
    #     crossover = torch.randint(0, b.shape[0], size=(1,))
    #     b[i, crossover:] = b2[i, crossover:]


def gen(t, fn, num_images=1, store=True):
    # Ensure the data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    torch.manual_seed(42)

    prompt_embeds, pooled_prompt_embeds = gen_prompt_embeds(t)
    
    images = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        guidance_scale=0.0,
        num_inference_steps=4,
        num_images_per_prompt=num_images,
    ).images

    torch.save(t, f"{data_dir}/{fn}.pt")
    for i, image in enumerate(images):
        f = f"{data_dir}/{fn}-{i}.jpg"
        image.save(f)
        yield f


def get_uuid():
    return str(uuid.uuid4())


def load(uuid):
    return torch.load("data/" + uuid + ".pt")


def delete_uuid(uuid):
    os.remove("data/" + uuid + ".pt")
    for f in glob.glob("data/" + uuid + "-*.jpg"):
        os.remove(f)


def recent():
    uuids = [os.path.basename(p).split(".")[0] for p in glob.glob("data/*.pt")]
    return [
        {"uuid": uuid, "images": list(glob.glob(f"data/{uuid}-*.jpg"))}
        for uuid in uuids
    ]
