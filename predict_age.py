import cv2

age_net = cv2.dnn.readNetFromCaffe(
    "models/deploy_age.prototxt", "models/age_net.caffemodel"
)
AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]

def predict_age(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746))
    age_net.setInput(blob)
    age_preds = age_net.forward()
    return AGE_BUCKETS[age_preds[0].argmax()]