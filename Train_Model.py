import cv2
import numpy as np
import os
from PIL import Image

path = "you"
recognizer = cv2.face.LBPHFaceRecognizer_create()



def getImageName(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    labels = []


    for imagePath in imagePaths:


        facePic = Image.open(imagePath).convert("L")
        faceNp = np.array(facePic, 'uint8')

        filename = os.path.basename(imagePath)
        parts = filename.split(".")

        label = int(parts[1])

        faces.append(faceNp)
        labels.append(label)

        cv2.imshow("training...", faceNp)
        cv2.waitKey(10)

    return faces, labels



faces, labels = getImageName(path)
recognizer.train(faces, np.array(labels))
recognizer.write("model.yml")
cv2.destroyAllWindows()


print("Faces:", len(faces), "Labels:", labels)
