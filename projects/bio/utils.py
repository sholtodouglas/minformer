import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorboardX import SummaryWriter


def visualize_token_prediction_difficulty(
    text, losses, save_image=True, writer: SummaryWriter | None = None, step: int | None = None
):
    # Clip losses between 0 and 10
    # TODO(sholto): Absolute numbers?
    clipped_losses = np.clip(losses, 0, 10)

    # Normalize clipped losses
    normalized_losses = clipped_losses / jnp.max(clipped_losses)

    # Define color gradient from green to red
    def get_color(value):
        r = int(255 * value)
        g = int(255 * (1 - value))
        return (r, g, 0)

    if save_image:
        # Create an image
        font_size = 20
        char_width = 10
        char_height = 20
        img_width = 1200
        chars_per_line = img_width // char_width
        num_lines = (len(text) - 1) // chars_per_line + 1
        img_height = num_lines * char_height

        img = Image.new("RGB", (img_width, img_height))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        for i, char in enumerate(text[1:], 1):
            color = get_color(normalized_losses[i - 1])
            x = ((i - 1) % chars_per_line) * char_width
            y = ((i - 1) // chars_per_line) * char_height
            # Draw text
            draw.text((x, y), char, fill=color, font=font)

        if writer is not None:
            if step is None:
                raise ValueError("Both 'writer' and 'step' must be provided to save to TensorBoard.")

            # Convert PIL image to numpy array
            img_array = np.array(img).transpose((2, 0, 1))  # HWC to CHW format

            # Add to TensorBoard
            writer.add_image("Token Prediction Difficulty", img_array, step)
            print(f"Wrote token difficulty viz at step {step}")
    else:
        for i, char in enumerate(text[1:], 1):
            color = get_color(normalized_losses[i - 1])
            print(f"\033[38;2;{color[0]};{color[1]};{color[2]}m{char}\033[0m", end="")
        print("\n")
