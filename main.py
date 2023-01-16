import gradio as gr
import numpy as np

from scripts.create_images_inversion import generate_images as generate_images_txt2img
from scripts.tf_hub import generate_images as generate_images_img2img


def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)


STYLE_MAPPING = {
    "Bonca 2D": ("embeddings/bonca-1.bin", (30, 9, "render")),
    "Bonca 1D": ("embeddings/bonca-2.bin", (30, 7, "portrait of an old man")),
    "Bonca 3D": ("embeddings/bonca-3.bin", (30, 6, "rendered image")),
}

dropdown_choices = list(STYLE_MAPPING.keys())


def generate_images(
    num_batches,
    style,
    prompt,
    num_images,
    num_inference_steps,
    guidance_scale,
    style_weight,
):
    embeddings = STYLE_MAPPING[style][0]
    return generate_images_txt2img(
        embeddings,
        prompt,
        num_images,
        num_batches,
        num_inference_steps,
        guidance_scale,
        style_weight,
    )


def txt2img_examples(style):
    return STYLE_MAPPING[style][1]


def generate_images_tf(img_1, img_2, style_weight, content_weight, num_steps):
    img_1 = img_1.astype("float32") / 255
    img_2 = img_2.astype("float32") / 255
    return generate_images_img2img(
        img_1, img_2, style_weight, content_weight, num_steps
    )


with gr.Blocks() as demo:
    with gr.Tab("text2img"):
        dropdown = gr.Dropdown(dropdown_choices, value=dropdown_choices[0])

        text = gr.Textbox(
            label="Enter your prompt",
            show_label=False,
            max_lines=1,
            placeholder="Enter your prompt",
        ).style(
            border=(True, False, True, True),
            rounded=(True, False, False, True),
            container=False,
        )
        num_inference_steps = gr.Slider(
            1, 100, value=30, step=1, label="Number of inference steps"
        )
        guidance_scale = gr.Slider(1, 20, value=7.5, step=0.5, label="Guidance scale")
        num_images = gr.Slider(1, 8, value=1, step=1, label="Number of images")
        num_batches = gr.Slider(1, 20, value=1, step=1, label="Number of batches")
        style_weight = gr.Slider(0, 3, value=1, step=0.1, label="Style weight")
        btn = gr.Button("Generate image").style(
            margin=False,
            rounded=(False, True, True, False),
        )
        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[3], height="auto")

    with gr.Tab("img2img"):
        with gr.Row():
            with gr.Column():
                image_style = gr.inputs.Image(shape=(512, 512))
                with gr.Accordion("Styles", open=False):
                    gr.Examples(
                        examples=[
                            "train_examples/train_1/0003.jpg",
                            "train_examples/train_1/0004.jpg",
                            "train_examples/train_1/0006.jpg",
                            "train_examples/train_2/0009.jpg",
                            "train_examples/train_2/0010.jpg",
                            "train_examples/train_3/0005.jpg",
                            "train_examples/train_3/0008.jpg",
                        ],
                        inputs=image_style,
                    )
            with gr.Column():
                image = gr.inputs.Image()
        num_steps = gr.Slider(1, 5000, value=2000, step=1, label="Number steps")
        style_weight_tf = gr.Slider(
            0, 0.0001, value=0.0001, step=0.000005, label="Style weight"
        )
        content_weight = gr.Slider(500, 2000, value=800, step=1, label="Content weight")
        btn_tf = gr.Button("Generate image").style(
            margin=False,
            rounded=(False, True, True, False),
        )
        gallery_tf = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[3], height="auto")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")

    btn.click(
        generate_images,
        inputs=[
            num_batches,
            dropdown,
            text,
            num_images,
            num_inference_steps,
            guidance_scale,
            style_weight,
        ],
        outputs=gallery,
    )

    btn_tf.click(
        generate_images_tf,
        inputs=[image_style, image, style_weight_tf, content_weight, num_steps],
        outputs=gallery_tf,
    )

    dropdown.change(
        txt2img_examples,
        inputs=dropdown,
        outputs=[num_inference_steps, guidance_scale, text],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
