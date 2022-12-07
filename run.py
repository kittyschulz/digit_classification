import os 
import cv2

from pipeline import classify, region_proposal_MSER
from vis import visualize_predictions

def identify_numbers(image_path, save, visualize=False, display_score=False):
    """
    Runs the pipeline and outputs an image with bounding box annotations.

    The pipeline consists of a region proposal step using MSER and a 
    classification step using a CNN. 

    Args:
        image_path (str): Path to the image to run the pipeline on.
        save (str): Path to save the output image to.
    
    Returns: 
        None. Saves an image to the specified path.
    """
    # Read the image.
    img = cv2.imread(image_path)
    # Resize the image if any dimension is larger than 800 px
    # This really helps with inference time in the region proposal stage.
    h, w, _ = img.shape
    if h > 800 or w > 800:
        if h > w:
            img = cv2.resize(img, (500,800))
        else:
            img = cv2.resize(img, (800,500))
    # Obtain region proposals from MSER.
    bboxes = region_proposal_MSER(img)
    # Apply gaussian filter.
    gauss = cv2.GaussianBlur(img.copy(),(5,5),cv2.BORDER_DEFAULT)
    # Classify regions.
    predictions = classify(gauss, bboxes, architecture='vgg+imagenet', confidence=0.98, nms_thresh=0.9)
    # Save the output. 
    visualize_predictions(img, predictions, visualize=visualize, display_score=display_score, save=save)

def main():
    INPUT_DIR = "example_images"
    OUTPUT_DIR = "graded_images"

    # Verify example images directory weights exist.
    if not os.path.exists(INPUT_DIR):
        raise Exception('Directory "./example_images" does not exist. Please download the example images here: https://gatech.box.com/s/24xvjqw8zuitltwesu1z5vttmf6c5l2e and make sure the folder is located within this directory.')

    # Make graded_images directory if it doesn't exist.
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    
    # Verify model weights exist.
    if not os.path.exists('./weights/vgg_imagenet.pth'):
        raise Exception('Model weights not found. Please make sure that the model weights are saved to "./weights". Model weights can be downloaded here: https://gatech.box.com/s/4hpnvcb1uwjc6tpw4395x8543jgsej0x')

    # Get list of images to run inference on.
    img_formats = [".jpg",".png",".jpeg",".bmp",".dib",".webp",".tiff",".tif",".sr",".ras"]
    image_files = [f for f in os.listdir('example_images') if os.path.isfile(os.path.join('example_images', f)) and f.endswith(tuple(img_formats))]

    if not image_files:
        raise Exception('No images were found in "./example_images". Please download the example images here: https://gatech.box.com/s/24xvjqw8zuitltwesu1z5vttmf6c5l2e.')

    for i, file in enumerate(image_files):
        print(f'Detecting and classifying digits in {os.path.basename(file)}.')
        try:
            identify_numbers(os.path.join(INPUT_DIR, file), save=os.path.join(OUTPUT_DIR, f'{i}.png'))
        except:
            print('No valid regions found.')
    
    print(f'{len(image_files)} images saved to directory "{OUTPUT_DIR}"')

if __name__ == "__main__":
    main()