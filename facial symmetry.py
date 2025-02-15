import cv2
import numpy as np

# Function to calculate facial symmetry based on edge orientation similarity (EOS)
def calculate_facial_symmetry(detector, landmarks, landmark1, landmark2, img): 
    # Calculate the midpoint between landmark1 and landmark2
    mid_point_x = int((landmarks.part(landmark1).x + landmarks.part(landmark2).x) / 2)

    # calling get frontal face detector, dlib
    faces = detector(img)
    
    # draw the bounding box for the face 
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.right(), face.bottom()
    
    # Restrict to the bounding box region
    img_rect = img[y:h, x:w]
    
    # Adjust midpoint relative to bounding box
    mid_point_x = mid_point_x - x  
    
    # splitting image into right and left part
    left_part = img_rect[:, :int(mid_point_x)]
    right_part = img_rect[:, int(mid_point_x):]
    
    # Ensure both parts have the same dimensions
    if left_part.shape[1] != right_part.shape[1]:
        min_width = min(left_part.shape[1], right_part.shape[1])
        left_part = left_part[:, :min_width]
        right_part = right_part[:, :min_width]
    
    # Sobel filter parameters
    scale = 3.0
    delta = 0
    depth = cv2.CV_16S

    # Left part of face
    left_gray = cv2.cvtColor(left_part, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(left_gray, depth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    gradient_y = cv2.Sobel(left_gray, depth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    # Right part of face
    right_gray = cv2.cvtColor(right_part, cv2.COLOR_BGR2GRAY)
    mirror_gradient_x = cv2.Sobel(right_gray, depth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    mirror_gradient_y = cv2.Sobel(right_gray, depth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    # Whole face gradients
    whole_face_gray = cv2.cvtColor(img_rect, cv2.COLOR_BGR2GRAY)
    whole_face_gradient_x = cv2.Sobel(whole_face_gray, depth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    whole_face_gradient_y = cv2.Sobel(whole_face_gray, depth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    # Apply Gaussian blur to smooth out the image
    whole_face_blurred = cv2.GaussianBlur(whole_face_gray, (5, 5), 0)
    right_blurred = cv2.GaussianBlur(right_gray, (5, 5), 0)
    left_blurred = cv2.GaussianBlur(left_gray, (5, 5), 0)
    
    # Difference between blurred image and original image 
    whole_face_difference = cv2.absdiff(whole_face_gray, whole_face_blurred)
    right_side_difference = cv2.absdiff(right_gray, right_blurred)
    left_side_difference = cv2.absdiff(left_gray, left_blurred)
    
    # Mean of differences
    face_smoothness = np.mean(whole_face_difference)
    right_side_smoothness = np.mean(right_side_difference)
    left_side_smoothness = np.mean(left_side_difference)
    
    # Smoothness score
    face_score = 1 - (face_smoothness / 255.0)
    face_score = max(0, min(face_score, 1))
    right_score = 1 - (right_side_smoothness / 255.0)
    right_score = max(0, min(right_score, 1))
    left_score = 1 - (left_side_smoothness / 255.0)
    left_score = max(0, min(left_score, 1))
    
    # Calculate the gradient for the images
    # Left side 
    abs_gradient_x = cv2.convertScaleAbs(gradient_x)
    abs_gradient_y = cv2.convertScaleAbs(gradient_y)
    gradient = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0) 
    
    # Right side 
    mirror_abs_gradient_x = cv2.convertScaleAbs(mirror_gradient_x)
    mirror_abs_gradient_y = cv2.convertScaleAbs(mirror_gradient_y)
    mirror_gradient = cv2.addWeighted(mirror_abs_gradient_x, 0.5, mirror_abs_gradient_y, 0.5, 0) 
    
    # Compute edge orientation (left and right side of face)
    theta = np.arctan2(gradient_y, gradient_x)
    theta_mirror = np.arctan2(mirror_gradient_y, mirror_gradient_x)
    
    # Compute angular difference (for left and right side of face)
    phi = np.abs(theta - theta_mirror)
    phi = np.minimum(phi, np.pi - phi)
    
    # Compute EOS (left and right)
    eos = 1 - (phi / np.pi)

    # Average EOS across all pixels (left and right)
    average_eos = round(np.mean(eos), 4)
    
    # Output smoothness and EOS scores
    print(f"face_score: {face_score}, right_score: {right_score}, left_score: {left_score}")

    return average_eos, face_score,left_score, right_score




