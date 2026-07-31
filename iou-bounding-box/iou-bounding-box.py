def box_area(box):
    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    return width * height

def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    intersection = []
    x1min = max(box_a[0], box_b[0])
    x2max = min(box_a[2], box_b[2])
    y1min = max(box_a[1], box_b[1])
    y2max = min(box_a[3], box_b[3])
    intersection.append(x1min)
    intersection.append(y1min)
    intersection.append(x2max)
    intersection.append(y2max)
    dientichgiao = float(box_area(intersection))
    dientichhop = float(box_area(box_a) + box_area(box_b) - dientichgiao)
    return dientichgiao / dientichhop
    