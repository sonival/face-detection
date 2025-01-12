import cv2
import cv2.data

def capture_video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao acessar camera de video.")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5, minSize=(30,30))

            #desenhando frame
            for(x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y), (x+w,y+h),(255,0,0),2)
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass 

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    capture_video()