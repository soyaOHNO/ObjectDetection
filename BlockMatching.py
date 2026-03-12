import cv2
import numpy as np

cap = cv2.VideoCapture(0)

template = cv2.imread("template.JPEG")
th, tw = template.shape[:2]

scales = np.linspace(0.01, 0.10, 10)

while True:

    if cv2.waitKey(1) == 27:
        break

    ret, frame = cap.read()

    best_val = 0
    best_loc = None
    best_scale = 1

    for scale in scales:

        resized = cv2.resize(template, None, fx=scale, fy=scale)

        h, w = resized.shape[:2]

        if h > frame.shape[0] or w > frame.shape[1]:
            continue

        result = cv2.matchTemplate(frame, resized, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale
            best_w = w
            best_h = h

    if best_val > 0.55:
        top_left = best_loc
        bottom_right = (top_left[0] + best_w, top_left[1] + best_h)
        cv2.rectangle(frame, top_left, bottom_right, (0,255,0),2)

    cv2.imshow("frame", frame)



cap.release()
cv2.destroyAllWindows()