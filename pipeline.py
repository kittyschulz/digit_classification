import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2

from model import load_model

def extract_regions(img, bboxes):
    """
    Extracts regions from an image given bounding boxes.

    Args:
        img (arr): image to extract the regions from.
        bboxes (list): a list of lists representing bounding boxes
        as [x0, y0, w, h].

    Returns:
        A normalized tensor of shape [n, 3, 32, 32] representing the extracted regions
    """
    # initiailize output tensor
    regions = np.zeros((len(bboxes), 3, 32, 32))
    for i, box in enumerate(bboxes):
        h, w, _ = img.shape
        # apply padding to regions based on region dimensions
        padding_x = max(box[3]//10, 5)
        padding_y = max(box[2]//10, 5)
        x_min = max(box[1] - padding_x, 0)
        y_min = max(box[0] - padding_y, 0)
        x_max = min(box[1]+box[3] + padding_x, h)
        y_max = min(box[0]+box[2] + padding_y, w)
        # crop the region from the image array
        crop = img[x_min:x_max, y_min:y_max, :]
        crop = cv2.resize(crop, (32, 32))
        crop = np.transpose(crop, (2, 0, 1))
        # add the cropped region to the output tensor
        regions[i, :] = crop
    # normalize the tensor
    return torch.tensor(regions/255.)


def filter_boxes(img, bboxes, ar_thresh=3, min_area_ratio=0.0004, max_area_ratio=0.04):
    """
    Filters region proposals.

    Args:
        img (arr): the image region proposals were obtained from
        bboxes (list): a list of lists representing bounding boxes
        as [x0, y0, w, h].
        ar_threshold (float): the aspect ratio threshold to use to filter the region proposals
        min_area_ratio (float): the minimum area threshold to filter regions based on their size 
        with respect to the image dimensions
        max_area_ratio (float): the maximum area threshold to filter regions based on their size 
        with respect to the image dimensions

    Returns:
        A list of the filtered region proposals
    """
    h, w, _ = img.shape
    min_area = min_area_ratio * h * w
    max_area = max_area_ratio * h * w
    final_boxes = []
    for box in bboxes:
        aspect_ratio = box[3]/box[2]
        area = box[2]*box[3]
        if aspect_ratio > 1 and aspect_ratio < ar_thresh:
            if area > min_area and area < max_area:
                final_boxes.append(box)
    return final_boxes


def region_proposal_MSER(img):
    """
    Runs MSER on an image to obtain region proposals.

    Args:
        img (arr): an image to run MSER on
    
    Returns:
        A list of bounding boxes representing the region proposals from MSER
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect regions in the grayscale image with MSER.
    mser = cv2.MSER_create(delta=10)
    _, bboxes = mser.detectRegions(gray)
    # Filter the regions by aspect ratio and minimum and maximum area.
    filtered_boxes = filter_boxes(img, bboxes)
    return filtered_boxes


def nms(predictions, iou_threshold=0.5):
    """
    Applys torchvision NMS to a set of predictions.

    Args:
        predictions (list): a list of dictionaries containing model predictions
        iou_threshold (float): the IoU threshold to use with NMS

    Returns:
        A list of predictions filtered by NMS
    """
    boxes = torch.FloatTensor([i['bbox'] for i in predictions])
    scores = torch.FloatTensor([i['score'] for i in predictions])
    idx = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold)
    return [predictions[i] for i in idx]


def classify(image, bboxes, architecture='cnn', confidence=0.9, nms_thresh=None):
    """
    Runs inference on regions of an image with the specified model.

    Args:
        img (arr): image to extract the regions from.
        bboxes (list): a list of lists representing bounding boxes
        as [x0, y0, w, h].
        architecture (str): a string specifying which model to use.
        confidence (float): the confidence threshold to apply to predictions
        nms_thresh (float): the IoU threshold to use with NMS. If None, NMS will not be applied.
    """
    # load model
    model = load_model(architecture)
    # extract regions
    regions = extract_regions(image, bboxes)
    outputs = model(regions.float())
    # apply Softmax to get probabilities
    outputs = F.softmax(outputs, -1)
    score, label = torch.max(outputs, 1)

    predictions = []
    # apply confidence threshold
    for i, b in enumerate(bboxes):
        if score[i] >= confidence and label[i] != 10:
            predictions.append({'bbox': [b[0], b[0]+b[2], b[1], b[1]+b[3]],
                                'label': label[i],
                                'score': score[i]})
    # apply NMS
    if nms_thresh:
        predictions = nms(predictions, iou_threshold=nms_thresh)
    return predictions