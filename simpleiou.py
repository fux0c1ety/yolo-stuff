import torch

def calculate_iou(boxA, boxB):
    #Extract bounding boxes coordinates
    x0_A, y0_A, x1_A, y1_A = boxA
    x0_B, y0_B, x1_B, y1_B = boxB
        
    # Get the coordinates of the intersection rectangle
    x0_I = max(x0_A, x0_B)
    y0_I = max(y0_A, y0_B)
    x1_I = min(x1_A, x1_B)
    y1_I = min(y1_A, y1_B)
    #Calculate width and height of the intersection area.
    width_I = x1_I - x0_I 
    height_I = y1_I - y0_I
    # Handle the negative value width or height of the intersection area
    #if width_I<0 : width_I=0
    #if height_I<0 : height_I=0
    width_I = torch.clamp(width_I, min=0)
    height_I = torch.clamp(height_I, min=0)
    # Calculate the intersection area:
    intersection = width_I * height_I
    # Calculate the union area:
    width_A, height_A = x1_A - x0_A, y1_A - y0_A
    width_B, height_B = x1_B - x0_B, y1_B - y0_B
    union = (width_A * height_A) + (width_B * height_B) - intersection
    # Calculate the IoU:
    IoU = intersection/union
    # for plotting purpose 
    boxI = torch.tensor([x0_I, y0_I, width_I,height_I])
    # Return the IoU and intersection box
    return IoU, boxI  
