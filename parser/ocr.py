import pytesseract
from PIL import Image
from typing import List
from models.pydantic_models import Token, BoundingBox

class OCRProcessor:
    @staticmethod
    def extract_text_with_bboxes(image: Image.Image) -> List[Token]:
        # Perform OCR with bounding boxes
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        tokens = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Filter out empty or low confidence detections
                text = data['text'][i].strip()
                if text:
                    bbox = BoundingBox(
                        x0=float(data['left'][i]),
                        y0=float(data['top'][i]),
                        x1=float(data['left'][i] + data['width'][i]),
                        y1=float(data['top'][i] + data['height'][i])
                    )
                    tokens.append(Token(
                        text=text,
                        bbox=bbox,
                        confidence=float(data['conf'][i]) / 100.0
                    ))
        return tokens
