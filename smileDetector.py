#SmileDetector

#Librerias de OpenCV
import cv2

#Carga los filtros creados en el XML
cara_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ojo_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

#Deteccion
def detect(gray, imagen):
    #La imagen original y la imagen en blanco y negro
    #1.3 factor de reduccion de la imagen
    #5 zonas deben ser aceptadas
    faces = cara_cascade.detectMultiScale(gray, 1.3, 5)
    #Coordenadas desde arriba a la izquierda x,y
    #W ancho del rectangulo
    #H altura del rectangulo
    for(x,y,w,h) in faces: 
        #Rectangle es una funcion dibuja el rectangulo
        #El tercer argumento son las coordenadas de la parte inferior izquierda
        #del rectangulo
        #2 es el ancho de la linea del rectangulo
        #imprime los rectangulos en la imagen original del video
        cv2.rectangle(imagen,(x,y),(x+w, y+h), (255,0,0), 2)
        #La zona del rectangulo para detectar los ojos para las dos imagenes
        #Coge la region de los ojos
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = imagen[y:y+h, x:x+w]
        
        eyes = ojo_cascade.detectMultiScale(roi_gray, 1.1, 2)
        
        for(ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh), (0,255,0), 2)
            cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh), (0,255,0), 2)
            #2 Ojos por eso dos rectangulos
            
        zona_gris = gray[y:y+h, x:x+w]
        zona_color = imagen[y:y+h, x:x+w]
        
        smile = smile_cascade.detectMultiScale(zona_gris, 1.5, 22)
        
        for(sx,sy,sw,sh) in smile: 
            cv2.rectangle(zona_color,(sx,sy),(sx+sw, sy+sh), (0,0,255), 2)
    
    return (imagen)#Devuelve la imagen con los rectangulos dibujados en la cara

#Trabajo con la webCam
#Coge el ultimo frame de la webCam
#0 si coges la webcam del ordenador 1 si es una webcam externa
video_captura = cv2.VideoCapture(0)
#Para todas las imagenes de la webCam
while True: #Bucle infinito hasta que pulsemos q
    #Para coger unoicamente el segundo parametro que nos da VideoCapture
    #que es el ultimo frame del video
    _,imagen = video_captura.read()
    #Para coger la imagen en escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    #Resultado de la funcion detect
    detectResult= detect(gray, imagen)
    #Muestra las imagenes tratadas por pantalla
    cv2.imshow('Video', detectResult)
    #Parar el bucle infinito
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_captura.release()
cv2.destroyAllWindows()
    
        