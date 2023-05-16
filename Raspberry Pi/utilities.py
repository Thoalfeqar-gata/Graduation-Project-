import mediapipe as mp, cv2, numpy as np

face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection = 1,
    min_detection_confidence = 0.5
)

def detect_faces_mp(image):
    h, w = image.shape[:2]
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.detections:
        return []
    dets = []
    for result in results.detections:
        if result.score[0] < 0.75:
            continue
        det = result.location_data.relative_bounding_box
        x, y, width, height = det.xmin, det.ymin, det.width, det.height
        x = int(x * w)
        y = int(y * h)
        width = int(width * w)
        height = int(height * h)
        dets.append([x, y, width, height])
    
    return dets