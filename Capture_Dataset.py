import cv2


dataSetPath = "Buffe_Jp"

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


recognizer = cv2.face.LBPHFaceRecognizer_create()


count = 0
limit = 100

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y ,w , h) in faces:
        if count >= limit:
            break

        count += 1
        face_img = gray[y:y + h, x:x + w]

        file_path = f"{dataSetPath}/muhka.{count}.jpg"
        cv2.imwrite(file_path, face_img)


        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("kinukuha", frame)



    cv2.imshow("Ako", frame)


    if cv2.waitKey(1) == ord("d"):
        break


cap.release()
cv2.destroyAllWindows()