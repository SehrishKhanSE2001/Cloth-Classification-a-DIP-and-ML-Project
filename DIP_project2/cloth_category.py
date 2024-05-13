
import cv2

# Load the images
shirt_img = cv2.imread('images/shirt.png', cv2.IMREAD_GRAYSCALE)
trouser_img = cv2.imread('images/trouser.png', cv2.IMREAD_GRAYSCALE)
sneaker_img = cv2.imread('images/sneaker.png', cv2.IMREAD_GRAYSCALE)
coat_img = cv2.imread('images/coat.png', cv2.IMREAD_GRAYSCALE)
dress_img = cv2.imread('images/dress.png', cv2.IMREAD_GRAYSCALE)
pullover_img = cv2.imread('images/pullover.jpg', cv2.IMREAD_GRAYSCALE)
sandal_img = cv2.imread('images/sandal.png', cv2.IMREAD_GRAYSCALE)
sweater_img = cv2.imread('images/sweater.png', cv2.IMREAD_GRAYSCALE)
bag_img = cv2.imread('images/bag.png', cv2.IMREAD_GRAYSCALE)
ankle_boot_img = cv2.imread('images/ankle_boot.png', cv2.IMREAD_GRAYSCALE)
jacket_img=cv2.imread('images/jacket.png',cv2.IMREAD_GRAYSCALE)
# Initialize the ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp_shirt, des_shirt = orb.detectAndCompute(shirt_img, None)
kp_trouser, des_trouser = orb.detectAndCompute(trouser_img, None)
kp_sneaker, des_sneaker = orb.detectAndCompute(sneaker_img, None)
kp_coat, des_coat = orb.detectAndCompute(coat_img, None)
kp_dress, des_dress = orb.detectAndCompute(dress_img, None)
kp_pullover, des_pullover = orb.detectAndCompute(pullover_img, None)
kp_sandal, des_sandal = orb.detectAndCompute(sandal_img, None)
kp_sweater, des_sweater = orb.detectAndCompute(sweater_img, None)
kp_bag, des_bag = orb.detectAndCompute(bag_img, None)
kp_ankle_boot, des_ankle_boot = orb.detectAndCompute(ankle_boot_img, None)
kp_jacket, des_jacket = orb.detectAndCompute(jacket_img, None)

# Initialize the BFMatcher (Brute Force Matcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def find_best_match(query_des, des_list):
    matches = bf.match(query_des, des_list)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[0].distance

# Test image
test_img = cv2.imread('new_image.png', cv2.IMREAD_GRAYSCALE)
kp_test, des_test = orb.detectAndCompute(test_img, None)

# Compare the test image with each reference image
distances = {
    "shirt": find_best_match(des_test, des_shirt),
    "trouser": find_best_match(des_test, des_trouser),
    "sneaker": find_best_match(des_test, des_sneaker),
    "coat": find_best_match(des_test, des_coat),
    "dress": find_best_match(des_test, des_dress),
    "pullover": find_best_match(des_test, des_pullover),
    "sandal": find_best_match(des_test, des_sandal),
    "sweater": find_best_match(des_test, des_sweater),
    "bag": find_best_match(des_test, des_bag),
    "ankle_boot": find_best_match(des_test, des_ankle_boot),
    "jacket": find_best_match(des_test, des_jacket),
}

# Set a threshold for matching distances
threshold = 200

# Check the distances and identify the object
for item, distance in distances.items():
    if distance < threshold:
        print(distance)
        print(f"The test image contains a {item}.")


