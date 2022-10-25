import cv2
import mediapipe as mp
import time

class faceDetect():

    def __init__(self, minDetectionCon = 0.75, modelSel = 0):
        self.minDetectionCon = minDetectionCon
        self.modelSel = modelSel

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon,
                                                                self.modelSel)
        self.mpDraw = mp.solutions.drawing_utils

    def findFace(self, frame):

        imRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imRGB)
        # print(results.detections)

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(frame, detection)
                # print(id, detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)

                # cv2.rectangle(frame, bbox, (255, 0, 255), 2)
                cv2.putText(frame, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        return frame, bbox

def main():
    cTime = 0
    pTime = 0

    cam = cv2.VideoCapture(0)
    detection = faceDetect()
    while True:
        Success, frame = cam.read()
        frame, b = detection.findFace(frame)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Face Detection", frame)
        key = cv2.waitKey(1)

        if key == 81 or key == 113:
            break


if __name__ == "__main__":
    main()