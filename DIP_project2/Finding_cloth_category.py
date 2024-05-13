import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tkinter.ttk as ttk

# this function identifies the category of the cloth in the image
def identify_object(test_image_path, reference_images): 
      # Load the images
    reference_data = {}
    orb = cv2.ORB_create()  # used for key point detection and feature matching in computer mapping
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #bruteforce matcher object

    for item, path in reference_images.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None) # kp abbreviation for key point. ORB identifies them as interesting location on the image
        reference_data[item] = {"kp": kp, "des": des} # des: descriptors. captures relevent info about key points and matches key points bwteen different images

    test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    kp_test, des_test = orb.detectAndCompute(test_img, None)

    distances = {}
    for item, data in reference_data.items():
        matches = bf.match(des_test, data["des"])
        matches = sorted(matches, key=lambda x: x.distance)
        distances[item] = matches[0].distance

    # Set a threshold for matching distances
    threshold = 200

    #Find the item with the smallest non-zero distance
   
    min_distance_item = min(((item, distance) for item, distance in distances.items() if 0.0 < distance < threshold), key=lambda x: x[1], default=None)




    # finding non zero distance item
    if min_distance_item:
        return f"The test image contains a {min_distance_item[0]}."
    else:
        return "No match found."

# Function 1: Remove Background
def remove_background(image, output_path, lower_threshold, upper_threshold):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array(lower_threshold, dtype=np.uint8)
    upper_bound = np.array(upper_threshold, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)
    result_image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(output_path, result_image)

# Function 2 Converting Image to Grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function 3 Applying Gaussian Blur
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Function 4 for applying canny edge detection
def apply_canny_edge_detection(image, threshold1=30, threshold2=100):
    return cv2.Canny(image, threshold1, threshold2)

# Function 5: Finding valid outermost contours
def find_valid_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []

    def is_contour_inside(contour, other_contour):
# x: The x-coordinate of the top-left corner of the bounding rectangle.
# y: The y-coordinate of the top-left corner of the bounding rectangle.
# w: The width of the bounding rectangle.
# h: The height of the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        x_other, y_other, w_other, h_other = cv2.boundingRect(other_contour)
        return x >= x_other and y >= y_other and x + w <= x_other + w_other and y + h <= y_other + h_other

    for i, contour in enumerate(contours):
        is_inside = False
        for j, other_contour in enumerate(contours):
            if i != j and is_contour_inside(contour, other_contour):
                is_inside = True
                break
        if not is_inside:
            valid_contours.append(contour)

    return valid_contours

# Function 6 for drawing and displaying contours
def draw_and_display_contours(image, contours, output_path='new_image.png'):
    black_background = np.zeros_like(image)
    cv2.drawContours(black_background, contours, -1, (255, 255, 255), thickness=cv2.FILLED) #The tuple (255, 255, 255) represents full intensity white, where each color channel is set to its maximum value (255).

    boundary_image = np.zeros_like(image)
    cv2.drawContours(boundary_image, contours, -1, (255, 255, 255), thickness=2)

    cv2.imshow('Valid Contours Boundary', boundary_image)
    cv2.imwrite(output_path, boundary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_ui():
    root = tk.Tk()
    root.title("Fashion Recognition App")

    style = ttk.Style()
    style.theme_use("clam")

    main_frame = ttk.Frame(root, padding=(20, 20, 20, 20))  # Increased padding for more space
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    title_label = ttk.Label(main_frame, text="Cloth Classification", font=("Helvetica", 20, "bold"))
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

    cloth_image = None

    def upload_image():
        nonlocal cloth_image
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            cloth_image = cv2.imread(file_path)
            show_image()

    def show_image():
        nonlocal cloth_image
        image = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((400, 400), Image.ANTIALIAS)  # Increased image size

        photo = ImageTk.PhotoImage(image)

        img_label = tk.Label(root, image=photo)
        img_label.image = photo
        img_label.pack(pady=10)

    def process_and_display_result():
        nonlocal cloth_image
        if cloth_image is not None:
            remove_background(cloth_image, "cloth_image.png", [0, 0, 0], [100, 100, 100])
            gray_image = convert_to_grayscale(cloth_image)
            blurred_image = apply_gaussian_blur(gray_image, kernel_size=(5, 5))
            edges = apply_canny_edge_detection(blurred_image, threshold1=30, threshold2=100)
            valid_contours = find_valid_contours(edges)
            draw_and_display_contours(cloth_image, valid_contours, output_path='new_image.png')

            reference_images = {
                # ... (reference images remain unchanged)
                 "shirt": 'images/shirt.png',
                "trouser": 'images/trouser.png',
    "sneaker": 'images/sneaker.png',
     "coat": 'images/coat.png',
    "dress": 'images/dress.png',
     "pullover": 'images/pullover.jpg',
     "sandal": 'images/sandal.png',
     "sweater": 'images/sweater.png',
     "bag": 'images/bag.png',
     "ankle_boot": 'images/ankle_boot.png',
    "jacket": 'images/jacket.png',

            }

            test_image_path = 'new_image.png'
            result = identify_object(test_image_path, reference_images)
            result_label.config(text=result)

    upload_button = ttk.Button(main_frame, text="Upload Image", command=upload_image, style="Round.TButton")
    upload_button.grid(row=2, column=0, pady=20, padx=10)

    process_button = ttk.Button(main_frame, text="Process and Display Result", command=process_and_display_result, style="Round.TButton")
    process_button.grid(row=2, column=1, pady=20, padx=10)

    result_label = ttk.Label(main_frame, text="", font=("Helvetica", 16))
    result_label.grid(row=3, column=0, columnspan=2, pady=(10, 0))

    root.mainloop()


if __name__ == "__main__":
    create_ui()