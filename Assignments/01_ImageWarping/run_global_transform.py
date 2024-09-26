import gradio as gr
import cv2
import numpy as np

max_translation = 300

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])


def extract_pxiel(image, x, y):
    if\
        x >= image.shape[0] or\
        y >= image.shape[1] or\
        x < 0 or\
        y < 0:
        return np.array((255,255,255))
    else:
        return image[x][y]

def interp_pixel(image, x, y):
    pos = [x, y]
    [left, bottom] = np.floor(pos).astype(int)
    [right, top] = [left + 1, bottom + 1]
    [digit_left, digit_bottom] = (pos - np.array([left, bottom])).astype(int)
    [digit_right, digit_top] = (np.array([1,1]) - np.array([digit_left, digit_bottom])).astype(int)
    left_top = extract_pxiel(image, left, top)
    right_top = extract_pxiel(image, right, top)
    left_bottom = extract_pxiel(image, left, bottom)
    right_bottom = extract_pxiel(image, right, bottom)
    weights = [digit_left * digit_top, digit_right * digit_top,\
            digit_left * digit_bottom, digit_right * digit_bottom]
    result = weights[0] * left_top +\
                weights[1] * right_top +\
                weights[2] * left_bottom +\
                weights[3] * right_bottom
    return result

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    full_shape = image.shape
    shape_array = np.array([full_shape[0], full_shape[1]])
    center = shape_array / 2
    rot_mat = np.array([[np.cos(rotation),  np.sin(rotation)],\
                        [-np.sin(rotation), np.cos(rotation)]])
    for i in range(full_shape[0]):
        for j in range(full_shape[1]):
            original_pos = np.array((i, j))
            offset = original_pos - center
            offset = offset - np.array([translation_x, translation_y])
            offset = offset / scale
            if flip_horizontal:
                offset = np.array([offset, full_shape[1] - offset - 1])
            offset = offset @ rot_mat
            original_pos = offset + center
            if\
                original_pos[0] >= full_shape[0] or\
                original_pos[1] >= full_shape[1] or\
                original_pos[0] < 0 or\
                original_pos[1] < 0:
                transformed_image[i][j] = np.array((255,255,255))
            else:
                transformed_image[i][j] = interp_pixel(image, *original_pos)



    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-max_translation, maximum=max_translation, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-max_translation, maximum=max_translation, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
