# Image Autoclassification Blur Detection Program based on CPBD
**Program name:** cpbd_image_blur_detection
**Authors:** Qingyuan Yang (UOM ID:11088331)
**Supervisor:** Fumie Costen
**Function:** The CPBD (Cumulative probability of blur detection) program is designed to detect image blur by analysing the contrast and edge information within blocks of an image. It is particularly useful for distinguishing between clear and blurred images.
              
## Program information
### Description
The CPBD program uses the Contrast-based Detection technique to analyse image blocks for edge information and contrast. It calculates the CPBD value for each block and determines whether the block is an edge block (indicating potential blur) or not. The program moves images into "clear image" or "blurred image" folders based on their CPBD values, allowing for automated sorting and categorization.
### Features
- Detects image blur using the CPBD algorithm.
- Automatically sorts images into "clear image" and "blurred image" folders.
- Provides accuracy assessment for sharpness and blur detection.

## How to run
### Environment
Windows system
Python 3.11 or higher
### Installation code
   ```
pip install opencv-python numpy matplotlib
   ```
### Usage
1. Place your images in the same directory with the program.
2. Run the program by executing the script:
   ```
   python cpbd_image_blur_detection.py
   ```
3. Enter the path to the directory containing your images.
4. The program will analyze the images using CPBD and move them into "clear image" or "blurred image" folders.

## Key Configuration
- `block_size`: The size of image blocks used for CPBD analysis. Default is 64x64 pixels.
- `threshold`: The threshold’ value for determining the clear or blur. It needs to be test through histogram analysis of test set’s CPBD value for more accuracy performance. In this case 0.09 for Method 1, 0.5 for Method 2.

## Dependencies
- Python 3.11
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

### How to label the test data
In this program, there is a accuracy test program for test dataset.
mark 'M' at the end of filename for blur image for example:'0_IPHONE-SE_M'
mark 'S' at the end of filename for sharp image for example:'0_IPHONE-SE_S'
