import cv2

class FaceDetector:
    def __init__(self, model_path=None):
        # Use OpenCV Haar cascade by default
        if model_path is None:
            model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(model_path)

    def detect_faces(self, image_bgr):
        """
        Detect faces in a BGR image.
        Returns list of bounding boxes [x, y, w, h].
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces


class FaceDetectorDNN:
    def __init__(self, model_path="res10_300x300_ssd_iter_140000.caffemodel",
                 config_path="deploy.prototxt"):
        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    def detect_faces(self, image_bgr, conf_threshold=0.5):
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(image_bgr, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                faces.append(box.astype(int))
        return faces



from mtcnn import MTCNN
import cv2

class FaceDetectorMTCNN:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, image_bgr):
        results = self.detector.detect_faces(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        faces = [r['box'] for r in results]
        return faces