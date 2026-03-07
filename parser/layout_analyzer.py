from typing import List
from models.pydantic_models import Page, Block, Line, Token, BoundingBox

class LayoutAnalyzer:
    @staticmethod
    def analyze_page_layout(page: Page, tokens: List[Token]) -> Page:
        # Simple heuristic: group tokens into lines based on Y-coordinate proximity
        if not tokens:
            return page

        # Sort tokens by top coordinate, then left coordinate
        sorted_tokens = sorted(tokens, key=lambda t: (t.bbox.y0, t.bbox.x0))
        
        lines = []
        if sorted_tokens:
            current_line_tokens = [sorted_tokens[0]]
            for i in range(1, len(sorted_tokens)):
                token = sorted_tokens[i]
                prev_token = current_line_tokens[-1]
                
                # Check if token is on the same line (heuristic: overlap in Y-axis)
                if abs(token.bbox.y0 - prev_token.bbox.y0) < 5: # 5px tolerance
                    current_line_tokens.append(token)
                else:
                    # Finalize current line
                    line = LayoutAnalyzer._create_line(current_line_tokens)
                    lines.append(line)
                    current_line_tokens = [token]
            
            # Finalize last line
            if current_line_tokens:
                lines.append(LayoutAnalyzer._create_line(current_line_tokens))

        # Group lines into a single block for now (can be improved with clustering)
        if lines:
            block_bbox = BoundingBox(
                x0=min(l.bbox.x0 for l in lines),
                y0=min(l.bbox.y0 for l in lines),
                x1=max(l.bbox.x1 for l in lines),
                y1=max(l.bbox.y1 for l in lines)
            )
            block = Block(
                type="text",
                bbox=block_bbox,
                lines=lines,
                text="\n".join(l.text for l in lines)
            )
            page.blocks = [block]

        return page

    @staticmethod
    def _create_line(tokens: List[Token]) -> Line:
        text = " ".join(t.text for t in tokens)
        bbox = BoundingBox(
            x0=min(t.bbox.x0 for t in tokens),
            y0=min(t.bbox.y0 for t in tokens),
            x1=max(t.bbox.x1 for t in tokens),
            y1=max(t.bbox.y1 for t in tokens)
        )
        return Line(text=text, bbox=bbox, tokens=tokens)
