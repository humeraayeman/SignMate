import cv2
from app.ml_service import predict_frame


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


print("Webcam started.")
print("Show different signs to the camera.")
print("Press Q to quit.")


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    result = predict_frame(frame)

    if result["success"]:
        text = (
            f'{result["label"]} '
            f'{result["confidence"] * 100:.1f}%'
        )
    else:
        text = result.get(
            "message",
            "No prediction"
        )

    cv2.putText(
        frame,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "SignSync ML Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()