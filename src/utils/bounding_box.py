# src/utils/bounding_box.py

from typing import Dict, List


def calculate_fixed_box(boxes: List[Dict[str, int]]) -> Dict[str, int]:
    """
    Calculates a single bounding box that represents the union of all provided boxes.

    This function is used to generate the "Fixed Box" output for a track, creating a
    single, static frame that encompasses the subject's entire range of motion
    within a scene.

    Args:
        boxes: A list of bounding box dictionaries, where each must have 'x', 'y',
               'width', and 'height' keys.

    Returns:
        A single dictionary representing the union of all input boxes.
    """
    if not boxes:
        return {}

    x_coords = [b['x'] for b in boxes]
    y_coords = [b['y'] for b in boxes]
    widths = [b['width'] for b in boxes]
    heights = [b['height'] for b in boxes]

    # Find the top-left corner of the union box
    x_min = min(x_coords)
    y_min = min(y_coords)

    # Find the bottom-right corner of the union box
    x_max = max(x + w for x, w in zip(x_coords, widths))
    y_max = max(y + h for y, h in zip(y_coords, heights))

    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }
