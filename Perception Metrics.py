import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image
import numpy as np
import dlib
import pandas as pd
from scipy.spatial import distance
import os
import sys
# import colorspacious as cs
# from matplotlib import colors as mcolor
from skimage import color,io 
from facial_symmetry import calculate_facial_symmetry

sys.path.append("/Users/user/Downloads/human perception with computer vision/Face-Recogntion-Detection/")
sys.path.append("nbrancati-py")


# Function adapted from https://github.com/wiseaidev/Face-Recogntion-Detection 
from skin_seg import Skin_Detect 


images = os.listdir("/Users/user/Downloads/human perception with computer vision/facesforexperiment2")
folder_path = '/Users/user/Downloads/human perception with computer vision/facesforexperiment2/'



# setting the detector and landmarks for measurements
detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor("/Users/user/Downloads/human perception with computer vision/shape_predictor_81_face_landmarks.dat")



image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
    if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(('.jpg', '.png'))]


for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    

# calculating distance and updating Dataframe

def calculate_distance_and_update_dataframe(landmarks, landmarks1, landmarks2, column_name, idx):
    x1,  y1 = landmarks.part(landmarks1).x, landmarks.part(landmarks1).y
    x2, y2 = landmarks.part(landmarks2).x, landmarks.part(landmarks2).y
    
    distance_value = distance.euclidean((x1,y1), (x2, y2))
    dataframe.at[idx, column_name] = distance_value
    

dataframe = pd.DataFrame({
    "Image Name": [None] * len(images),
    'Face Height': [0.0] * len(images),
    'Face Height 2': [0.0] * len(images),
    'Face Width': [0.0] * len(images),
    'Face Width 2': [0.0] * len(images),
    'Orbits Intercanthal Width': [0.0] * len(images),
    'Orbits Fissure Length(left)': [0.0] * len(images),
    'Orbits Fissure Length(right)': [0.0] * len(images),
    'Orbits Biocular Width': [0.0] * len(images),
    'Nose Height': [0.0] * len(images),
    'Nose Width': [0.0] * len(images),
    'Labio-oral region': [0.0] * len(images),
    'Intercanthal Face Height': [0.0] * len(images),
    'eye fissure height (left)': [0.0] * len(images),
    'eye fissure height (right)': [0.0] * len(images),
    'orbit and brow height (right)': [0.0] * len(images),
    'orbit and brow height (left)': [0.0] * len(images),
    'Columella Length': [0.0] * len(images),
    'Upper Lip Height': [0.0] * len(images),
    'Lower Vermilion Height': [0.0] * len(images),
    'Philtrum Width': [0.0] * len(images),
    'lateral upper lip heights (left)': [0.0] * len(images),
    'lateral upper lip heights (right)': [0.0] * len(images),
    'Lower Face Height': [0.0] * len(images),
    'Upper Vermilion Height': [0.0] * len(images)
})

# Define the measurement columns for further processing
measurement_columns = [
    'Face Height', 'Face Height 2',
    'Face Width', 'Face Width 2',
    'Orbits Intercanthal Width', 'Orbits Fissure Length(left)', 'Orbits Fissure Length(right)', 'Orbits Biocular Width',
    'Nose Height', 'Nose Width', 'Labio-oral region', 'Intercanthal Face Height',
    'eye fissure height (left)', 'eye fissure height (right)', 'orbit and brow height (right)', 'orbit and brow height (left)',
    'Columella Length', 'Upper Lip Height', 'Lower Vermilion Height', 'Philtrum Width',
    'lateral upper lip heights (left)', 'lateral upper lip heights (right)', 'Lower Face Height', 'Upper Vermilion Height'
]



dataframe = pd.DataFrame({col: [np.nan] * len(images) for col in measurement_columns})
dataframe['Image Name'] = [None] * len(images)

for idx, im in enumerate(images):
    try:
        # Skip DS_Store
        if im == "DS_Store":
            continue
        
        # Read and check the image
        image_path = '/Users/user/Downloads/human perception with computer vision/facesforexperiment2/' + im
        image = cv2.imread(image_path)
        
        # Check if image loaded successfully
        if image is None:
            print(f"Unable to read image {im}")
            dataframe.loc[idx, measurement_columns] = np.nan  
            continue  

        # Store the image name in the dataframe
        dataframe.at[idx, "Image Name"] = im
        
        # Convert the image to RGB color space and detect faces
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_image)
        
        # Ensure that at least one face is detected before proceeding
        if len(faces) == 0:
            print(f"No face detected in image {im}")
            dataframe.loc[idx, measurement_columns] = np.nan 
            continue
        
        # Get the first face detected
        face = faces[0]
        landmarks = landmarks_predictor(rgb_image, face)
        
        # Perform distance calculations and update the DataFrame
        # calculate_distance_and_update_dataframe(landmarks, 71, 27, 'Head Height', idx)
        # calculate_distance_and_update_dataframe(landmarks, 71, 8, 'Face Height', idx)
        # calculate_distance_and_update_dataframe(landmarks, 27, 8, 'Face Height 2', idx)
        # calculate_distance_and_update_dataframe(landmarks, 33, 8, 'Face Height 3', idx)
        # calculate_distance_and_update_dataframe(landmarks, 2, 14, 'Face Width', idx)
        # calculate_distance_and_update_dataframe(landmarks, 5, 11, 'Face Width 2', idx)
        # calculate_distance_and_update_dataframe(landmarks, 39, 42, 'Orbits Intercanthal Width', idx)
        # calculate_distance_and_update_dataframe(landmarks, 42, 36, 'Orbits Fissure Length(left and right)', idx)
        # calculate_distance_and_update_dataframe(landmarks, 36, 45, 'Orbits Biocular Width', idx)
        # calculate_distance_and_update_dataframe(landmarks, 27, 33, 'Nose Height', idx)
        # calculate_distance_and_update_dataframe(landmarks, 31, 35, 'Nose Width', idx)
        # calculate_distance_and_update_dataframe(landmarks, 48, 54, 'Labio-oral region', idx)
        # calculate_distance_and_update_dataframe(landmarks, 27, 68, 'Intercanthal Face Height', idx)
        # calculate_distance_and_update_dataframe(landmarks, 38, 41, 'eye fissure height (left and right)', idx)
        # calculate_distance_and_update_dataframe(landmarks, 19, 41, 'orbit and brow height (left and right)', idx)
        # calculate_distance_and_update_dataframe(landmarks, 33, 30, 'Columella Length', idx)
        # calculate_distance_and_update_dataframe(landmarks, 33, 66, 'Upper Lip Height', idx)
        # calculate_distance_and_update_dataframe(landmarks, 68, 57, 'Lower Vermilion Height', idx)
        # calculate_distance_and_update_dataframe(landmarks, 49, 53, 'Philtrum Width', idx)
        # calculate_distance_and_update_dataframe(landmarks, 32, 62, 'lateral upper lip heights (left and right)', idx)
        # # distance between stomion and gnathion
        # calculate_distance_and_update_dataframe(landmarks, 66, 8, 'Lower Face Height', idx)
        # # distance between stomion and libiale superious
        # calculate_distance_and_update_dataframe(landmarks, 51, 68, 'Upper Vermilion Height', idx)
        
        calculate_distance_and_update_dataframe(landmarks, 27, 8, 'Face Height', idx)
        calculate_distance_and_update_dataframe(landmarks, 33, 8, 'Face Height 2', idx)
        calculate_distance_and_update_dataframe(landmarks, 3, 13, 'Face Width', idx)
        calculate_distance_and_update_dataframe(landmarks, 5, 11, 'Face Width 2', idx)
        calculate_distance_and_update_dataframe(landmarks, 39, 42, 'Orbits Intercanthal Width', idx)
        calculate_distance_and_update_dataframe(landmarks, 39, 36, 'Orbits Fissure Length(left)', idx)
        calculate_distance_and_update_dataframe(landmarks, 42, 45, 'Orbits Fissure Length(right)', idx)
        calculate_distance_and_update_dataframe(landmarks, 36, 42, 'Orbits Biocular Width', idx)
        calculate_distance_and_update_dataframe(landmarks, 27, 33, 'Nose Height', idx)
        calculate_distance_and_update_dataframe(landmarks, 31, 35, 'Nose Width', idx)
        calculate_distance_and_update_dataframe(landmarks, 48, 54, 'Labio-oral region', idx)
        calculate_distance_and_update_dataframe(landmarks, 27, (51 + 57) // 2,'Intercanthal Face Height', idx)
        calculate_distance_and_update_dataframe(landmarks, 37, 41, 'eye fissure height (left)', idx)
        calculate_distance_and_update_dataframe(landmarks, 44, 46, 'eye fissure height (right)', idx)
        calculate_distance_and_update_dataframe(landmarks, 24, 46, 'orbit and brow height (right)', idx)
        calculate_distance_and_update_dataframe(landmarks, 19, 41,'orbit and brow height (left)', idx)
        calculate_distance_and_update_dataframe(landmarks, 33, 30, 'Columella Length', idx)
        calculate_distance_and_update_dataframe(landmarks, 33, (51 + 57) // 2, 'Upper Lip Height', idx)
        calculate_distance_and_update_dataframe(landmarks, 57, (51 + 57) // 2,'Lower Vermilion Height', idx)
        calculate_distance_and_update_dataframe(landmarks, 49, 53,'Philtrum Width', idx)
        calculate_distance_and_update_dataframe(landmarks, 32, 51,'lateral upper lip heights (left)', idx)
        calculate_distance_and_update_dataframe(landmarks, 34, 51, 'lateral upper lip heights (right)', idx)
        # distance between stomion and gnathion
        calculate_distance_and_update_dataframe(landmarks, 6, 10,'Lower Face Height', idx)
        # distance between stomion and libiale superious
        calculate_distance_and_update_dataframe(landmarks, 8, (51 + 57) // 2,'Upper Vermilion Height', idx)
    except Exception as e:
        print(f"Error processing image {im}: {str(e)}")
        dataframe.loc[idx, measurement_columns] = np.nan 
        continue  # Continue to the next image in case of an error

# Show the DataFrame
pd.set_option('display.max_columns', None)

dataframe['Facial index'] = dataframe['Face Height 2'] / dataframe['Face Width']
dataframe['Mandibular index'] = dataframe['Lower Face Height'] / dataframe['Face Width 2']
dataframe['Intercanthal index'] = dataframe['Orbits Intercanthal Width'] / dataframe['Orbits Biocular Width']
dataframe['Orbital width index (left and right)'] = (dataframe['Orbits Fissure Length(left)'] + dataframe['Orbits Fissure Length(right)']) / dataframe['Orbits Intercanthal Width']
dataframe['Eye fissure index (left and right)'] = (dataframe['eye fissure height (left)'] + dataframe['eye fissure height (right)']) / (dataframe['Orbits Fissure Length(left)'] + dataframe['Orbits Fissure Length(right)'])
dataframe['Nasal index'] = dataframe['Nose Width'] / dataframe['Nose Height']
dataframe['Vermilion height index'] = dataframe['Upper Vermilion Height'] / dataframe['Lower Vermilion Height']
dataframe['Mouth-face width index'] = dataframe['Labio-oral region'] / dataframe['Face Width']

pd.set_option('display.max_columns', None)




eos_average = []
face_smoothness = []
left_side_smoothness = []
right_side_smoothness = []
average_ita_values = []
image_name = []

for image_file in image_files:
    try:
        image_path = os.path.join(folder_path, image_file)
        
        your_image = cv2.imread(image_path)
        
        img = cv2.imread(image_path)
        
        
        faces = detector(img)
        
        if faces:
            face = faces[0]
            
            landmarks = landmarks_predictor(rgb_image,face)
            
            results= calculate_facial_symmetry(detector,landmarks,39,42, img)
            # print(results)
            eos_average.append(results[0])
            face_smoothness.append(results[1])
            left_side_smoothness.append(results[2])
            right_side_smoothness.append(results[3])
            
            
        else:
            print(f"Error: Could not load image at {image_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        pass
    
    
    skin_detector = Skin_Detect()
    

    _image = cv2.imread(image_path)
    
    
    # Resize the image to reduce memory usage
    _image = cv2.resize(_image, (500, 500))  # Resize to 500x500 pixels
    resulting_image = skin_detector.RGB_H_CbCr(_image)
    skin_mask, seg = resulting_image
    
    lab_img = color.rgb2lab(seg.astype('uint8'))

    
    ita_values = []

    for pixel in lab_img.reshape((-1, 3)):
        L, a, b = pixel

        if L > 0:  
            ita = np.arctan((L - 50) / b) * (180 / np.pi)
            ita_values.append(ita)

    average_ita = np.mean(ita_values)


    average_ita_values.append(average_ita)
    
    
    

if len(eos_average) < len(dataframe):
    eos_average.extend([np.nan] * (len(dataframe) - len(eos_average)))
elif len(eos_average) > len(dataframe):
    eos_average = eos_average[:len(dataframe)]
dataframe['edge_of_similarity_average'] = eos_average


if len(face_smoothness) < len(dataframe):
    face_smoothness.extend([np.nan] * (len(dataframe) - len(face_smoothness)))
elif len(face_smoothness) > len(dataframe):
    face_smoothness = face_smoothness[:len(dataframe)]
dataframe['face_smoothness'] = face_smoothness


if len(left_side_smoothness) < len(dataframe):
    left_side_smoothness.extend([np.nan] * (len(dataframe) - len(left_side_smoothness)))
elif len(left_side_smoothness) > len(dataframe):
    left_side_smoothness= left_side_smoothness[:len(dataframe)]
dataframe['left_side_smoothness'] = left_side_smoothness


if len(right_side_smoothness) < len(dataframe):
    right_side_smoothness.extend([np.nan] * (len(dataframe) - len(right_side_smoothness)))
elif len(right_side_smoothness) > len(dataframe):
    right_side_smoothness = right_side_smoothness[:len(dataframe)]
dataframe['right_side_smoothness'] = right_side_smoothness


# Check if lengths match before adding to dataframe
if len(average_ita_values) < len(dataframe):
    average_ita_values.extend([np.nan] * (len(dataframe) - len(average_ita_values)))
elif len(average_ita_values) > len(dataframe):
    average_ita_values = average_ita_values[:len(dataframe)]

dataframe['Average_ITA'] = average_ita_values

file_path = "/Users/user/Downloads/human perception with computer vision/v10 face perception metrics.csv"

dataframe.to_csv(file_path, index = False) 


# individual typology angle
# average_ita_values = []


# for image_file in image_files:
    
#     image_path = os.path.join(folder_path, image_file)
    
#     skin_detector = Skin_Detect()
    

#     _image = cv2.imread(image_path)
    
    
#     # Resize the image to reduce memory usage
#     _image = cv2.resize(_image, (500, 500))  # Resize to 500x500 pixels
#     resulting_image = skin_detector.RGB_H_CbCr(_image)
#     skin_mask, seg = resulting_image
    
#     lab_img = color.rgb2lab(seg.astype('uint8'))

    
#     ita_values = []

#     for pixel in lab_img.reshape((-1, 3)):
#         L, a, b = pixel

#         if L > 0:  
#             ita = np.arctan((L - 50) / b) * (180 / np.pi)
#             ita_values.append(ita)

#     average_ita = np.mean(ita_values)


#     average_ita_values.append(average_ita)

# # Check if lengths match before adding to dataframe
# if len(average_ita_values) < len(dataframe):
#     average_ita_values.extend([np.nan] * (len(dataframe) - len(average_ita_values)))
# elif len(average_ita_values) > len(dataframe):
#     average_ita_values = average_ita_values[:len(dataframe)]

# dataframe['Average_ITA'] = average_ita_values

# print(dataframe.info())
# print(dataframe)

# file_path = "/Users/user/Downloads/human perception with computer vision/v3 face perception metrics.csv"

# dataframe.to_csv(file_path, index = False) 