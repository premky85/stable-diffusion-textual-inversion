import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)


def load_learned_embed_in_clip(
    learned_embeds_path, text_encoder, tokenizer, token=None
):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer."
        )

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds


def generate_images(
    embeddings_path: str,
    prompt: str,
    num_images: int,
    num_batches: int = 1,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    style_weight: float = 1,
):
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    load_learned_embed_in_clip(embeddings_path, text_encoder, tokenizer)
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    ).to("cuda")

    token = "<bonca-style>"
    prompt = f"{prompt}, in the style of ({token}:{style_weight})"
    images = []
    for i in range(num_batches):
        images += pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
        ).images
    torch.cuda.empty_cache()
    return images
    # for i in range(num_images):
    #     filename = prompt.replace(token, "").replace("  ", " ").replace(" ", "_").replace(",", "")
    #     images[i].save(f"./out/{filename}_{i:05d}.png")
