import cv2

# Load Pretrained MobileNet SSD Model
prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(
    prototxt,
    model
)

# Object Classes
classes = [
    "background", "aeroplane", "bicycle", "bird",
    "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Start Webcam
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Resize Frame
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    # Detect Objects
    net.setInput(blob)

    detections = net.forward()

    # Loop Through Detections
    for i in range(detections.shape[2]):

        confidence =
            detections[0, 0, i, 2]

        # Minimum Confidence
        if confidence > 0.5:

            idx =
                int(detections[0, 0, i, 1])

            label = classes[idx]

            box =
                detections[0, 0, i, 3:7] * \
                [width, height, width, height]

            (startX, startY, endX, endY) =
                box.astype("int")

            # Draw Rectangle
            cv2.rectangle(
                frame,
                (startX, startY),
                (endX, endY),
                (0, 255, 0),
                2
            )

            # Display Label
            text =
                f"{label}: {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Show Output
    cv2.imshow(
        "Object Detection System",
        frame
    )

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
