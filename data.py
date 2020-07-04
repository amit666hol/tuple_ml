import cv2
import numpy as np
cap = cv2.VideoCapture(0)
min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)
x, y, rad = 20, 20, 20
r, g, b = 255,255,255
pos = True
x_forward, y_forward = 3, 20
color_forward = 5
while True:
    img = cv2.flip(cap.read()[1], 1)
    image_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(image_YCrCb, min_YCrCb, max_YCrCb)
    frame = cv2.bitwise_and(img, img, mask = skinRegionYCrCb)
    # Change the circle
    if b == 0:
        # Save the image, must be done before circle drawing
        name = "tuple_data\\{}.png".format(str(x) + "_" + str(y))
        cv2.imwrite(name, frame)
        if x >= 640 - rad or x < rad:
            pos = not pos
            y += y_forward
        if pos:
            x += x_forward
        else:
            x -= x_forward
    else:
        b -= color_forward
        g -= color_forward
    # Draw circle
    cv2.circle(frame, (x, y), rad, (r, g, b), -1)
    cv2.imshow("A", frame)

    if cv2.waitKey(2) & 0xFF == ord("w"):
        break