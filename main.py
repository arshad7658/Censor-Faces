face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap  = cv2.VideoCapture(0)
clahe = cv2.createCLAHE(2,(5,5))
while True:
    success, frame = cap.read()
    if not success:
        break
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahed = clahe.apply(grey_frame)
    bbox = face_cascade.detectMultiScale(clahed, 1.1, 8, minSize = (30,30))
    if not len(bbox):
        comp = cv2.GaussianBlur(frame, (99,99), 30)
    for x,y,w,h in bbox:
        roi = clahed[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, (99,99), 30)
        bgr_roi = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        frame[y:y+h, x:x+w, :] = bgr_roi
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
