#test
from deepface import DeepFace
import cv2

#Loading the image
image_path = 'img2.jpg'
image = cv2.imread(image_path)

#Checking if the image was loaded successfully
if image is None:
    print("Error, image not loaded")
    exit()

#analyze image for emotions

try:
    res = DeepFace.analyze(image, actions=['emotion'])
    
    # Print the full analysis result first
    print("\nFull analysis result:")
    print(res)
    
    # Find the highest emotion confidence across all faces
    highest_emotion = (None, 0)
    for face in res:
        emotions = face['emotion']
        dominant = max(emotions.items(), key=lambda x: x[1])
        if dominant[1] > highest_emotion[1]:
            highest_emotion = dominant
    
    # Print the dominant emotion
    print("\nDominant emotion analysis:")
    if highest_emotion[1] > 50:  # 50% confidence threshold
        print(f"Dominant emotion: {highest_emotion[0]} ({highest_emotion[1]:.2f}%)")
    else:
        print("No dominant emotion detected with high confidence")

except Exception as e:
    print(f"Error: {e}")

print("End of code")