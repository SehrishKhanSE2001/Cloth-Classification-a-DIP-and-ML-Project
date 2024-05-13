
# import cv2
# import numpy as np

# def load_image(image_path):
#     return cv2.imread(image_path)

# def remove_background(image, output_path, lower_threshold, upper_threshold):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     lower_bound = np.array(lower_threshold, dtype=np.uint8)
#     upper_bound = np.array(upper_threshold, dtype=np.uint8)
#     mask = cv2.inRange(hsv, lower_bound, upper_bound)
#     mask = cv2.bitwise_not(mask)
#     result_image = cv2.bitwise_and(image, image, mask=mask)
#     cv2.imwrite(output_path, result_image)
#     return result_image

# def preprocess_image(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
#     edges = cv2.Canny(blurred_image, threshold1=30, threshold2=100)
#     return edges

# def find_valid_contours(edges):
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     valid_contours = []
#     for i, contour in enumerate(contours):
#         is_inside = False
#         for j, other_contour in enumerate(contours):
#             if i != j and is_contour_inside(contour, other_contour):
#                 is_inside = True
#                 break
#         if not is_inside:
#             valid_contours.append(contour)
#     return valid_contours

# def draw_valid_contours(image, contours, output_path):
#     black_background = np.zeros_like(image)
#     cv2.drawContours(black_background, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
#     cv2.imwrite(output_path, black_background)
#     return black_background

# def draw_contours_boundary(image, contours, output_path):
#     boundary_image = np.zeros_like(image)
#     cv2.drawContours(boundary_image, contours, -1, (255, 255, 255), thickness=2)
#     cv2.imwrite(output_path, boundary_image)
#     return boundary_image

# def is_contour_inside(contour, other_contour):
#     x, y, w, h = cv2.boundingRect(contour)
#     x_other, y_other, w_other, h_other = cv2.boundingRect(other_contour)
#     return x >= x_other and y >= y_other and x + w <= x_other + w_other and y + h <= y_other + h_other

# def main():
#     # Load the cloth image
#     cloth_image = load_image('sandals.jpg')

#     # Function 1: Remove background
#     remove_background(cloth_image, "cloth_image.png", [0, 0, 0], [100, 100, 100])

#     # Function 2: Preprocess image
#     edges = preprocess_image(cloth_image)

#     # Function 3: Find valid contours
#     valid_contours = find_valid_contours(edges)

#     # Function 4: Draw valid contours filled
#     draw_valid_contours(cloth_image, valid_contours, 'filled_contours.png')

#     # Function 5: Draw contours boundary
#     draw_contours_boundary(cloth_image, valid_contours, 'boundary_image.png')

#     # Function 6: Display and save the outline of the valid contours
#     cv2.imshow('Valid Contours Boundary/ Defined Shape', draw_contours_boundary(cloth_image, valid_contours, 'new_image.png'))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np

# Function 1: Remove Background
def remove_background(image, output_path, lower_threshold, upper_threshold):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array(lower_threshold, dtype=np.uint8)
    upper_bound = np.array(upper_threshold, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.bitwise_not(mask)
    result_image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite(output_path, result_image)

# Function 2: Convert Image to Grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function 3: Apply Gaussian Blur
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Function 4: Apply Canny Edge Detection
def apply_canny_edge_detection(image, threshold1=30, threshold2=100):
    return cv2.Canny(image, threshold1, threshold2)

# Function 5: Find Valid Outermost Contours
def find_valid_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []

    def is_contour_inside(contour, other_contour):
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

# Function 6: Draw Contours and Display
def draw_and_display_contours(image, contours, output_path='new_image.png'):
    black_background = np.zeros_like(image)
    cv2.drawContours(black_background, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    boundary_image = np.zeros_like(image)
    cv2.drawContours(boundary_image, contours, -1, (255, 255, 255), thickness=2)

    cv2.imshow('Valid Contours Boundary', boundary_image)
    cv2.imwrite(output_path, boundary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example Usage
cloth_image = cv2.imread('PDAY.jpg')

# Function 1: Remove Background
remove_background(cloth_image, "cloth_image.png", [0, 0, 0], [100, 100, 100])

# Function 2: Convert Image to Grayscale
gray_image = convert_to_grayscale(cloth_image)

# Function 3: Apply Gaussian Blur
blurred_image = apply_gaussian_blur(gray_image, kernel_size=(5, 5))

# Function 4: Apply Canny Edge Detection
edges = apply_canny_edge_detection(blurred_image, threshold1=30, threshold2=100)

# Function 5: Find Valid Outermost Contours
valid_contours = find_valid_contours(edges)

# Function 6: Draw Contours and Display
draw_and_display_contours(cloth_image, valid_contours, output_path='new_image.png')


