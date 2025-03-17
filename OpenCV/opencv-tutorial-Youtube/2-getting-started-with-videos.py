import cv2

cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cap.set(3, 1208)
# cap.set(4, 720)
print(cap.get(3), cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter("outputgray.avi", fourcc, 20.0, (640, 480))
while True:
    ret, frame = cap.read()
    if ret: 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(gray)
        cv2.imshow("frame", gray)
        if cv2.waitKey(1) == ord("q"):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
