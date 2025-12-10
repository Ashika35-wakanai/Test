import cv2
from numpy.ma.core import negative

cap = cv2.VideoCapture(0)
faceFront = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")



while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frontFaceLine = faceFront.detectMultiScale(gray, 1.3, 5)


    faces = list(frontFaceLine)
    for (x, y ,w , h) in faces:
        number, conf = recognizer.predict(gray[y:y+h, x:x+w])

        positive = f"{"you:"}  {conf}%"
        negative = f"{"Unknown"} {conf}%"
        if conf < 60:
            cv2.putText(frame, positive, (x, y - 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x,y), (x + w, y+ h), (0, 165, 255), 5)

        else:
            cv2.putText(frame, negative, (x, y - 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)


    cv2.imshow("Ako", frame)


    if cv2.waitKey(1) == ord("d"):
        break


cap.release()
cv2.destroyAllWindows()

