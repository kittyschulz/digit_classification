import cv2
import matplotlib.pyplot as plt
from pipeline import filter_boxes, classify

def visualize_predictions(img, predictions, display_score=True, visualize=False, save=False):
    """
    Generates an image with annotated bounding boxes.

    Args:
        img (arr): The image to annotate.
        predictions (list): A list of dictionaries containing predicted bounding boxes.
        display_score (bool): A boolean indicating whether the label score should be annotated on the boxes.
        visualize (bool): A boolean indicating whether a plot should be displayed. 
        save (bool, str): Path to save the annotated image. If False, image will not be saved.
    """
    colors = {0:(0,255,0), 
            1:(255,128,0),
            2:(0,0,255),
            3:(255,255,0),
            4:(0,255,255),
            5:(255,0,255),
            6:(255,0,255),
            7:(0,128,255),
            8:(128,0,255),
            9:(255,0,128)}
    for i in predictions:
        x0, x1, y0, y1 = int(i['bbox'][0]), int(i['bbox'][1]), int(i['bbox'][2]), int(i['bbox'][3])
        if display_score:
            text = f'{int(i["label"].item())}: {round(i["score"].item()*100)}'
        else:
            text = f'{int(i["label"].item())}'
        cv2.rectangle(img, (x0, y0), (x1, y1), color=colors[int(i["label"].item())], thickness=2) 
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)   
        img = cv2.rectangle(img, (x0, y0-20), (x0+w, y0), colors[int(i["label"].item())], -1)
        img = cv2.putText(img, text, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    if save:
        cv2.imwrite(save, img)
    if visualize:
        plt.imshow(img[:,:,::-1])


def generate_output_video(input_video, output_video, model='vgg+imagenet'):
    """
    Generates a video with bounding box annotations.

    Args:
        input_video (str): Path to the input video.
        output_video (str): Path where output video should be saved.
        model (str): A string representing which model to run inference with. 

    Returns:
        None. Saves a video to the specified path.
    """
    # Video dimensions and FPS are just set as global variables here to help runtime. 
    W, H = 800, 500
    FPS = 30
    # Read video.
    cap = cv2.VideoCapture(input_video)
    # Initialize output video.
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MP4V'), FPS, (W,H))
    while True:
        ret, img = cap.read()
        if not ret:
            break
        # Resizing the video to 800 x 500 really helps with inference time. (It will
        # take > 1 housr run if a full frame video is passed!)
        img = cv2.resize(img, (W, H))
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Obtain region proposals
        mser = cv2.MSER_create(delta=10)
        _, bboxes = mser.detectRegions(gray)
        # Filter region proposals
        filtered_boxes = filter_boxes(img, bboxes, ar_thresh=3, min_area_ratio=0.0025, max_area_ratio=0.5)
        # Apply gaussian filter
        gauss = cv2.GaussianBlur(img.copy(),(7,7),cv2.BORDER_DEFAULT)
        # Run inference
        predictions = classify(gauss, filtered_boxes, confidence=0.98, architecture=model, nms_thresh=None)
        # Annotate video frame
        out_frame = visualize_predictions(img, predictions, visualize=False, save=False, display_score=False)
        # Write frame to output video
        out.write(out_frame)
    # Release output video
    out.release()