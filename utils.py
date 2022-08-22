import dlib, cv2

def get_cascades():
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')
    
    return face_cascade, eye_cascade, smile_cascade

def get_detector_predictor():
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('dlib data/shape_predictor_68_face_landmarks.dat')
    
    return face_detector, predictor

def extract_face_and_modalities_cv(path):
    
    face_cascade, eye_cascade, smile_casacde = get_cascades()
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 13)
    if len(faces) <= 0:
        print(f"opencv hasn't found any faces in image ({path}).")
        return None
    
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        face_roi_gray = gray[y : y + h, x : x + w]
        face_roi = img[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor = 1.1, minNeighbors = 10)
        smiles = smile_casacde.detectMultiScale(face_roi_gray, scaleFactor = 1.1, minNeighbors = 10)
        
        for eye in eyes:
            ex, ey, ew, eh = eye
            cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        
        for smile in smiles:
            sx, sy, sw, sh = smile
            cv2.rectangle(face_roi, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    
    return img    


def extract_face_and_modalities_dlib(path, preprocessing = True, clahe_then_modality = True, image_number = 0):
    
    detector, predictor = get_detector_predictor()
    img = cv2.imread(path)
    detections = detector(img, 1)
    
    if len(detections) <= 0:
        print(f"dlib hasn't found any faces in image ({path}).")
        return None
    
    #used to make the eyes, mouth, and nose rectangles bigger
    img_w, img_h = img.shape[:2]
    fraction = 0.008
    padding_x, padding_y = int(img_w * fraction), int(img_h * fraction)
    
    #for preprocessing
    clahe = cv2.createCLAHE(2)
    
    modalities = []
    detection_counter = 0
    for d in detections:
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        marks = predictor(img, d)
        
        #left eye
        lex1, ley1, lex2, ley2 = marks.part(36).x, marks.part(37).y, marks.part(39).x, marks.part(41).y
        
        #right eye
        rex1, rey1, rex2, rey2 = marks.part(42).x, marks.part(43).y, marks.part(45).x, marks.part(47).y

        #mouth
        mx1, my1, mx2, my2 = marks.part(48).x, marks.part(50).y, marks.part(54).x, marks.part(57).y

        #nose
        nx1, ny1, nx2, ny2 = marks.part(31).x, marks.part(27).y, marks.part(35).x, marks.part(33).y

        order = ['left eye', 'right eye', 'mouth', 'nose']
        #if preprocessing is required
        if preprocessing:
            if clahe_then_modality:
                roi_padding = 0
                ROI = img[y1 - roi_padding : y2 + roi_padding, x1 - roi_padding : x2 + roi_padding].copy()  
                ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                ROI = cv2.GaussianBlur(ROI, (3, 3), 0)
                ROI = clahe.apply(ROI)
                
                displacement_x = roi_padding - x1
                displacement_y = roi_padding - y1
                
                find = lambda x1, x2, y1, y2: ROI[y1 - padding_y + displacement_y : y2 + padding_y + displacement_y, x1 - padding_x + displacement_x : x2 + padding_x + displacement_x]
                
                left_eye = find(lex1, lex2, ley1, ley2)
                right_eye = find(rex1, rex2, rey1, rey2)
                mouth = find(mx1, mx2, my1, my2)
                nose = find(nx1, nx2, ny1, ny2)
                
                m = [left_eye, right_eye, mouth, nose]
                count = 0
                for modality in m:
                    cv2.imwrite(f'data/modalities/{order[count]} {image_number}__{detection_counter}.jpg', modality)
                    count += 1
                cv2.imwrite(f'data/preprocessed/{image_number}__{detection_counter}.jpg', ROI)
                modalities.append(m)
                    
            else:
                left_eye =  img[ley1 - padding_y : ley2 + padding_y, lex1 - padding_x : lex2 + padding_x]
                right_eye =  img[rey1 - padding_y : rey2 + padding_y, rex1 - padding_x : rex2 + padding_x]
                mouth =  img[my1 - padding_y : my2 + padding_y, mx1 - padding_x : mx2 + padding_x]
                nose =  img[ny1 - padding_y : ny2 + padding_y, nx1 - padding_x : nx2 + padding_x]
                
                m = []
                count = 0
                for modality in [left_eye, right_eye, mouth, nose]:
                    temp = cv2.cvtColor(modality, cv2.COLOR_BGR2GRAY)
                    temp = cv2.GaussianBlur(temp, (3, 3), 0)
                    temp = clahe.apply(temp)
                    m.append(temp)
                    cv2.imwrite(f'data/modalities/extract modality then preprocess {order[count]} {detection_counter}__{image_number}.jpg', cv2.resize(temp, (0, 0), fx = 10, fy = 10))
                    count += 1
                modalities.append(m)
                
        
       
        detection_counter += 1
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.rectangle(img, (lex1 - padding_x, ley1 - padding_y), (lex2 + padding_x, ley2 + padding_y), (255, 0, 0), 2)
        # cv2.rectangle(img, (rex1 - padding_x, rey1 - padding_y), (rex2 + padding_x, rey2 + padding_y), (255, 0, 0), 2)
        # cv2.rectangle(img, (mx1 - padding_x, my1 - padding_y), (mx2 + padding_x, my2 + padding_y), (0, 0, 255), 2)        
        # cv2.rectangle(img, (nx1 - padding_x, ny1 - padding_y), (nx2 + padding_x, ny2 + padding_y), (255, 255, 0), 2)

    return img