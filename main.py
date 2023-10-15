import cv2
from ultralytics import YOLO
import supervision as sv

def main():
    cap =cv2.VideoCapture("predict/444.mp4")
    model = YOLO("best.pt")
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret,frame =cap.read()
        if not ret:
            break

        results = model(frame, conf=0.85)[0]
        detections = sv.Detections.from_ultralytics(results)
        newFrame = box_annotator.annotate(scene=frame, detections=detections)

        cv2.imshow("video", frame)
        if cv2.waitKey(30) == 27:
            break


if __name__=="__main__":
    main()
