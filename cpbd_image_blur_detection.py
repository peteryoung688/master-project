import os
import cv2
import numpy as np
import shutil
import time
import matplotlib.pyplot as plt

def cpbd(image_path,block_size,indicator):
    # read and convert image from RGB to grayscale
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # divide to 64x64 squares and perform edge detection
    prob_blur_values = []
    # get the index of each block, (rows, cols)
    rows = gray_image.shape[0] // block_size
    cols = gray_image.shape[1] // block_size
    # matrix for contract to width_jnb, contract<51 width_jnb=5, contract>51 width_jnb=3
    width_jnb = np.concatenate((5 * np.ones(51), 3 * np.ones(205)))

    # edge detection in each block
    for i in range(rows):
        for j in range(cols):
            x = j * block_size
            y = i * block_size
            # define the starting point of the chosen block
            block = gray_image[y:y + block_size, x:x + block_size]
            # using canny kernel for ED, 100 and 200 represent the lower and upper thresholds
            edges = cv2.Canny(block, 100, 200)
            # count edge pixels. if it exceed 2%, its edge block, otherwise not
            if np.count_nonzero(edges) >= 0.002 * (block_size ** 2):
                if np.random.rand() > 0.5:  # randomly decide whether to process the block
                    contrast = np.max(block) - np.min(block)  # get the contrast of the block
                    blk_jnb = width_jnb[contrast]  # get the block W_jnb according to its contrast
                    widths = []
                    # randomly decide process row or col
                    if np.random.rand() > 0.5:
                        edges = zip(*edges)  # transpose the edge matrix
                    # determine the edge width
                    for line in edges:
                        current_width = 0
                        max_width = 0
                        # find the longest continuous pixels
                        for num in line:
                            if num != 0:
                                current_width += 1
                                max_width = max(max_width, current_width)
                            else:
                                current_width = 0

                        widths.append(max_width)

                    edge_width = sum(widths) / len(widths)

                    # Calculate the probability of blur detection at the edges
                    prob_blur = 1 - np.exp(-np.abs(edge_width / blk_jnb) ** 3.6)
                    prob_blur_values.append(prob_blur)

    if indicator == "1":
        # Method 1
        probability = np.mean(prob_blur_values)

    elif indicator == "2":
        # Method 2
        data = [x if x >= 0.01 else np.nan for x in prob_blur_values]
        less_than_threshold = [x for x in data if x < 0.64]
        if len(data) > 0:
            probability = len(less_than_threshold) / len(data)
        else:
            probability = 0

    return probability


def create_directories(path):
    clear_folder = os.path.join(path, "clear image")
    os.makedirs(clear_folder, exist_ok=True)

    blurred_folder = os.path.join(path, "blurred image")
    os.makedirs(blurred_folder, exist_ok=True)

    print(f"'clear image' and 'blurred image' folders have been created in {path}.")

def load_file(path, file_type):
    output_folder = os.path.join(path, "pending_image")
    os.makedirs(output_folder, exist_ok=True)
    # get the specific file type from the file_path
    files = os.listdir(path)
    files_of_type = [file for file in files if file.endswith(file_type)]

    for file in files_of_type:
        source_file = os.path.join(path, file)
        destination_file = os.path.join(output_folder, file)
        shutil.copy(source_file, destination_file)

    print(f" {file_type} files in {path} have been copied to the 'pending_image'.")

def image_move(cpbd_value, file_path,threshold):
    parent = os.path.dirname(os.path.dirname(file_path))
    if cpbd_value > threshold: # 0.5 is an experiment obtained thershold for blur and clear CPBD value, it may change in
        # different dataset. For further use this value can be obtained precisely through ML method
        destination_folder = os.path.join(parent, "clear image")
    else:
        destination_folder = os.path.join(parent, "blurred image")

    shutil.move(file_path, destination_folder)
    print(f"Image moved to: {destination_folder}")

def accuracy_test(file_path):
    # the accuracy assessment program is designed specifically for this dataset since the label is in the image file name
    # initial all the count number
    clear_s_count = 0
    clear_m_count = 0
    blurred_s_count = 0
    blurred_m_count = 0
    clear_folder = os.path.join(file_path, 'clear image')
    blurred_folder = os.path.join(file_path, 'blurred image')

    for filename in os.listdir(clear_folder):
        if filename.split('.')[0].endswith("S"):
            clear_s_count += 1
        elif filename.split('.')[0].endswith("M"):
            clear_m_count += 1

    for filename in os.listdir(blurred_folder):
        if filename.split('.')[0].endswith('S'):
            blurred_s_count += 1
        elif filename.split('.')[0].endswith('M'):
            blurred_m_count += 1

    if (blurred_s_count + clear_s_count)+(blurred_m_count + clear_m_count)!=0:
        print("Accuracy of Sharpness detection is:", clear_s_count / (blurred_s_count + clear_s_count)*100,"%")
        print("Accuracy of Blur detection is:", blurred_m_count / (blurred_m_count + clear_m_count)*100,"%")

def main(block_size):
    # main loop for the AUTO-image_selection
    file_path = input("Please enter the path to the file: ")
    print("File path entered:", file_path)

    while True: #  starts an infinite loop
        indicator = input("Please enter the method to use: ")
        print("Method choose entered:", indicator)
        if indicator == "1":
            threshold = 0.09
            break
        elif indicator == "2":
            threshold = 0.5
            break
        else:
            print("Wrong input, choose Method 1 or 2")

    # file_path = "test"
    create_directories(file_path)
    load_file(file_path, ('.JPG', '.jpg', '.jpeg'))
    image_path = os.path.join(file_path, "pending_image")

    start = time.perf_counter()
    total_images = len(os.listdir(image_path))
    processed_images = 0
    cpbd_values = []
    # Process each file in the given path
    for file_name in os.listdir(image_path):
        # Construct the full path to the file
        this_image_path = os.path.join(image_path, file_name)
        # Process the image using the CPBD function
        cpbd_value = cpbd(this_image_path,block_size,indicator)
        cpbd_values.append(cpbd_value)
        processed_images += 1
        # Print the image path and its corresponding CPBD value
        percentage_completion = (processed_images / total_images) * 100
        print(f"Image: {this_image_path},CPBD value: {cpbd_value},"
              f"Percentage completion: {round(percentage_completion,2)}%")
        image_move(cpbd_value, this_image_path,threshold)
    shutil.rmtree(os.path.join(file_path, "pending_image"))
    accuracy_test(file_path)
    end = time.perf_counter()
    elapsed = end - start
    print(f"Total Time:{round(elapsed,2)}s,average time per image:{round(elapsed/total_images,2)}s per image")

    plt.hist(cpbd_values, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histogram of CPBD Values')
    plt.xlabel('CPBD Value')
    plt.ylabel('Frequency')
    plt.show()

# the initial value for CPBD according to D.Narvekar's paper, block size of 64*64 and beta value 3.6. the value may
# change in different situation
block_size = 64
main(block_size)

