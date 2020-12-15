#Reconoce el rostro usando la cámara de la laptop
import cv2

cap = cv2.VideoCapture(0)
clasificador = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    imagenGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = clasificador.detectMultiScale(imagenGris, 1.3, 8)

    for (x,y,w,h) in rostros:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
