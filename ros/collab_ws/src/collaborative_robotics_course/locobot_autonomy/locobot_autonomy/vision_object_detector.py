#!/usr/bin/env python3

from PIL import Image
import io
from google.cloud import vision
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np
import cv2

class VisionObjectDetector:
    def __init__(self):
        """
        Initialize the Vision client once during object creation.
        """
        self.client = vision.ImageAnnotatorClient()

    def find_center_vertices(self, image_bytes, object_name):
        image = vision.Image(content=image_bytes)
        response = self.client.object_localization(image=image)
        objects = response.localized_object_annotations

        for obj in objects:
            if obj.name.lower() == object_name.lower():
                vertices = obj.bounding_poly.normalized_vertices
                x = (vertices[0].x + vertices[2].x) / 2
                y = (vertices[0].y + vertices[2].y) / 2

                # Convert normalized coordinates to pixels
                image_pil = PILImage.open(io.BytesIO(image_bytes))
                image_width, image_height = image_pil.size

                return (int(x * image_width), int(y * image_height)), vertices

        return None, []  # Return safe default values if no object found


    def annotate_image(self, vertices, im):
        # Convert NumPy array to PIL Image
        if isinstance(im, np.ndarray):
            pil_image = PILImage.fromarray(im)
        else:
            pil_image = PILImage.open(im)  # Only for cases where im is a file-like object

        image_width, image_height = pil_image.size
        draw = ImageDraw.Draw(pil_image)

        # Convert normalized vertices to pixel coordinates
        pixel_vertices = [(v.x * image_width, v.y * image_height) for v in vertices]

        # Draw bounding polygon
        draw.polygon(pixel_vertices, outline='red', width=3)

        # Add label near the first vertex
        if pixel_vertices:
            font_size = max(10, image_width // 50)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            draw.text(pixel_vertices[0], "Object", fill='red', font=font)

        return pil_image  # Return the annotated image


    # def annotate_image(self, image_bytes):
    #     """
    #     Detects all objects in the image and returns a PIL Image with
    #     bounding boxes and labels drawn for each detected object.

    #     :param image_bytes: The raw bytes of the image.
    #     :return: A PIL Image object annotated with bounding boxes and labels.
    #     """

    #     # Step 1: Create the Vision Image object from bytes
    #     image = vision.Image(content=image_bytes)
    #     # Step 2: Send the image to the API for object localization
    #     response = self.client.object_localization(image=image)
    #     # Step 3: Extract localized object annotations
    #     objects = response.localized_object_annotations
    #     # Step 4: Open the image via PIL for drawing
    #     pil_image = PILImage.open(io.BytesIO(image_bytes))
    #     image_width, image_height = pil_image.size
    #     # Step 5: Iterate through all the objects and draw the bounding boxes on the image.
    #     # Hint: draw.polygon allows you to draw based on pixel coordinates
    #     draw = ImageDraw.Draw(pil_image)
    #     for obj in objects:
    #         vertices = obj.bounding_poly.normalized_vertices
    #         x0 = vertices[0].x * pil_image.width
    #         y0 = vertices[0].y * pil_image.height
    #         x1 = vertices[2].x * pil_image.width
    #         y1 = vertices[2].y * pil_image.height
    #         draw.rectangle([x0, y0, x1, y1], outline='red', width=3)
    #         draw.text((x0, y0), obj.name, fill='red', font=ImageFont.truetype("arial.ttf", image_width // 50))
    #     #Step 6: Return the annotated image
    #     return pil_image

