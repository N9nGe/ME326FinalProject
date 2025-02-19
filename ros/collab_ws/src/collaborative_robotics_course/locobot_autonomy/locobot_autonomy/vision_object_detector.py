#!/usr/bin/env python3

from PIL import Image
import io
from google.cloud import vision
from PIL import Image as PILImage, ImageDraw, ImageFont

class VisionObjectDetector:
    def __init__(self):
        """
        Initialize the Vision client once during object creation.
        """
        self.client = vision.ImageAnnotatorClient()

    def find_center(self, image_bytes, object_name):
        """
        Finds the center of an object (e.g., "pineapple") in the provided image bytes.

        :param image_bytes: The raw bytes of the image.
        :param object_name: The target object name to search for (case-insensitive).
        :return: Tuple (pixel_x, pixel_y) of the object's approximate center, or None if not found.
        """

        # Step 1: Create the Vision Image object from bytes
        image = vision.Image(content=image_bytes)
        # Step 2: Send the image to the API for object localization
        response = self.client.object_localization(image=image)
        # Step 3: Extract localized object annotations
        objects = response.localized_object_annotations
        # Step 4: Search for the specified object. Hint: Objects returns all detected objects
        for obj in objects:
            if obj.name.lower() == object_name.lower():
                break
        else:
            return None
        # Step 5: Once the object is found, determine the position from the bounding box. Hint: obj.bounding_poly returns the bounding box
        vertices = obj.bounding_poly.normalized_vertices
        # Step 6: Find the center from the corners of the bounding box
        x = (vertices[0].x + vertices[2].x) / 2
        y = (vertices[0].y + vertices[2].y) / 2
        # Step 7: Return the center in pixel coordinates. Hint: The position of the bounding box is normalized so you will need to convert it back into the dimensions of the image
        image_pil = Image.open(io.BytesIO(image_bytes))
        image_width, image_height = image_pil.size  # Get width and height
        return (int(x * image_width), int(y * image_height))

    def annotate_image(self, image_bytes):
        """
        Detects all objects in the image and returns a PIL Image with
        bounding boxes and labels drawn for each detected object.

        :param image_bytes: The raw bytes of the image.
        :return: A PIL Image object annotated with bounding boxes and labels.
        """

        # Step 1: Create the Vision Image object from bytes
        image = vision.Image(content=image_bytes)
        # Step 2: Send the image to the API for object localization
        response = self.client.object_localization(image=image)
        # Step 3: Extract localized object annotations
        objects = response.localized_object_annotations
        # Step 4: Open the image via PIL for drawing
        pil_image = PILImage.open(io.BytesIO(image_bytes))
        image_width, image_height = pil_image.size
        # Step 5: Iterate through all the objects and draw the bounding boxes on the image.
        # Hint: draw.polygon allows you to draw based on pixel coordinates
        draw = ImageDraw.Draw(pil_image)
        for obj in objects:
            vertices = obj.bounding_poly.normalized_vertices
            x0 = vertices[0].x * pil_image.width
            y0 = vertices[0].y * pil_image.height
            x1 = vertices[2].x * pil_image.width
            y1 = vertices[2].y * pil_image.height
            draw.rectangle([x0, y0, x1, y1], outline='red', width=3)
            draw.text((x0, y0), obj.name, fill='red', font=ImageFont.truetype("arial.ttf", image_width // 50))
        #Step 6: Return the annotated image
        return pil_image

