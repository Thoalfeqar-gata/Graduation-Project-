import dlib, cv2, os, numpy as np, csv, face_recognition, skimage.util as util, time
from tqdm import tqdm
from keras_facenet import FaceNet
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import SpectralClustering
from descriptors.LocalDescriptors import WeberPattern, LocalBinaryPattern
import tensorflow as tf
import mediapipe as mp


def get_cascades():
    face_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_smile.xml')
    
    return face_cascade, eye_cascade, smile_cascade

def get_detector_predictor():
    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/dlib data/shape_predictor_68_face_landmarks.dat')
    
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
                    cv2.imwrite(f'data/modalities/{order[count]}/{image_number}__{detection_counter}.jpg', modality)
                    count += 1
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

def show_descriptors_performance():
    samples =  745
    path = 'data/modalities/'

    size = (320, 320)

    left_eyes = [os.path.join(path, x) for x in os.listdir(path) if 'left eye' in x]
    right_eyes = [os.path.join(path, x) for x in os.listdir(path) if 'right eye' in x]
    mouths = [os.path.join(path, x) for x in os.listdir(path) if 'mouth' in x]
    noses = [os.path.join(path, x) for x in os.listdir(path) if 'nose' in x]
    faces = [os.path.join('data/preprocessed', x) for x in os.listdir('data/preprocessed')]
    negatives = [os.path.join('data/negative/img', x) for x in os.listdir('data/negative/img')]

    train_labels_total = []
    train_data_total = []

    clusters = 50
    params = dict(algorithm = 1, trees = 5)
    extractors = [cv2.SIFT_create(), cv2.xfeatures2d.SURF_create()]

    for extractor in extractors:
        train_labels = []
        train_data = []
        
        matcher = cv2.FlannBasedMatcher(params, {})
        BOWKmeans = cv2.BOWKMeansTrainer(clusters)
        BOWExtractor = cv2.BOWImgDescriptorExtractor(extractor, matcher)
        
        for i in tqdm(range(int(samples * 0.25))):
            try:
                left_eye = cv2.resize(cv2.imread(left_eyes[i], cv2.IMREAD_GRAYSCALE), size)
                right_eye = cv2.resize(cv2.imread(right_eyes[i], cv2.IMREAD_GRAYSCALE), size)
                mouth = cv2.resize(cv2.imread(mouths[i], cv2.IMREAD_GRAYSCALE), size)
                nose = cv2.resize(cv2.imread(noses[i], cv2.IMREAD_GRAYSCALE), size)
                
                # face = cv2.resize(cv2.imread(faces[i], cv2.IMREAD_GRAYSCALE), size)
                # negative = cv2.resize(cv2.imread(negatives[i], cv2.IMREAD_GRAYSCALE), size)
                
                _, descriptor_left_eye = extractor.compute(left_eye, extractor.detect(left_eye))
                _, descriptor_right_eye = extractor.compute(right_eye, extractor.detect(right_eye))
                _, descriptor_mouth = extractor.compute(mouth, extractor.detect(mouth))
                _, descriptor_nose = extractor.compute(nose, extractor.detect(nose))
                
                BOWKmeans.add(descriptor_left_eye)
                BOWKmeans.add(descriptor_right_eye)
                BOWKmeans.add(descriptor_mouth)
                BOWKmeans.add(descriptor_nose)
                
                # _, descriptor_face = extractor.compute(face, extractor.detect(face))
                # _, descriptor_negative = extractor.compute(negative, extractor.detect(negative))
                # BOWKmeans.add(descriptor_face)
                # BOWKmeans.add(descriptor_negative)
            except:
                pass

        vocabulary = BOWKmeans.cluster()
        BOWExtractor.setVocabulary(vocabulary)


        for i in tqdm(range(samples)):
            
            left_eye = cv2.resize(cv2.imread(left_eyes[i], cv2.IMREAD_GRAYSCALE), size)
            right_eye = cv2.resize(cv2.imread(right_eyes[i], cv2.IMREAD_GRAYSCALE), size)
            mouth = cv2.resize(cv2.imread(mouths[i], cv2.IMREAD_GRAYSCALE), size)
            nose = cv2.resize(cv2.imread(noses[i], cv2.IMREAD_GRAYSCALE), size)
            
            # face = cv2.resize(cv2.imread(faces[i], cv2.IMREAD_GRAYSCALE), size)
            # negative = cv2.resize(cv2.imread(negatives[i], cv2.IMREAD_GRAYSCALE), size)
            
            try:
                features_left_eye = BOWExtractor.compute(left_eye, extractor.detect(left_eye))
                features_right_eye = BOWExtractor.compute(left_eye, extractor.detect(right_eye))
                features_mouth = BOWExtractor.compute(left_eye, extractor.detect(mouth))
                features_nose = BOWExtractor.compute(left_eye, extractor.detect(nose))
                
                if(not np.any(features_left_eye)):
                    continue
                elif(not np.any(features_right_eye)):
                    continue
                elif(not np.any(features_mouth)):
                    continue
                elif(not np.any(features_nose)):
                    continue

                train_data.append(features_left_eye.ravel())
                train_data.append(features_right_eye.ravel())
                train_data.append(features_mouth.ravel())
                train_data.append(features_nose.ravel())
                train_labels.append(1)
                train_labels.append(2)
                train_labels.append(3)
                train_labels.append(4)
                
                # features_face = BOWExtractor.compute(face, extractor.detect(face))
                # features_negative = BOWExtractor.compute(negative, extractor.detect(negative))
                
                # if(not np.any(features_face)):
                #     continue
                # elif(not np.any(features_negative)):
                #     continue
                
                # train_data.append(features_face.ravel())
                # train_data.append(features_negative.ravel())
                # train_labels.append(1)
                # train_labels.append(2)
            except:
                pass
        
        train_labels_total.append(train_labels)
        train_data_total.append(train_data)
            


    local_descriptors = [WeberPattern().compute, LocalBinaryPattern(24, 8).compute, hog]

    for local in local_descriptors:
            
        train_labels = []
        train_data = []
        for i in tqdm(range(samples)):
            left_eye = cv2.resize(cv2.imread(left_eyes[i], cv2.IMREAD_GRAYSCALE), size)
            right_eye = cv2.resize(cv2.imread(right_eyes[i], cv2.IMREAD_GRAYSCALE), size)
            mouth = cv2.resize(cv2.imread(mouths[i], cv2.IMREAD_GRAYSCALE), size)
            nose = cv2.resize(cv2.imread(noses[i], cv2.IMREAD_GRAYSCALE), size)

            # face = cv2.resize(cv2.imread(faces[i], cv2.IMREAD_GRAYSCALE), size)
            # negative = cv2.resize(cv2.imread(negatives[i], cv2.IMREAD_GRAYSCALE), size)
            
            fd_1 = local(left_eye)
            fd_2 = local(right_eye)
            fd_3 = local(mouth)
            fd_4 = local(nose)
            
            train_labels.append(1)
            train_labels.append(2)
            train_labels.append(3)
            train_labels.append(4)
            
            train_data.append(fd_1)
            train_data.append(fd_2)
            train_data.append(fd_3)
            train_data.append(fd_4)
            
            # fd_1 = local(face)
            # fd_2 = local(negative)
            
            # train_data.append(fd_1)
            # train_data.append(fd_2)
            # train_labels.append(1)
            # train_labels.append(2)
        
        train_data_total.append(train_data)
        train_labels_total.append(train_labels)


    descriptors = ['SIFT', 'SURF',  'Weber', 'LBP', 'HOG']
    modalities = ['left eye', 'right eye', 'mouth', 'nose']
    line_styles = [':', '-', '--', '-.']
    colors = ['darkorange', 'forestgreen', 'aqua', 'salmon']
    i = 0
    for train_data, train_labels in zip(train_data_total, train_labels_total):
        
        train_labels = label_binarize(train_labels, classes = [1, 2, 3, 4])
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size = 0.25, train_size = 0.75)
        mlp = MLPClassifier((196, 256, 128), max_iter = 10000)
        mlp.fit(X_train, y_train)

        y_pred = mlp.predict(X_test)
        y_true = y_test
        y_score = mlp.predict_proba(X_test)
        
        if i < 3:
            plt.subplot(3, 3, i + 1)
        else:
            plt.subplot(2, 2, i)
            
        for j in range(len(modalities)):
            fpr, tpr, _ = roc_curve(y_true[:, j], y_score[:, j])
            AUC = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw = 2, color = colors[j], linestyle = line_styles[j], label = f'ROC curve for {modalities[j]} with AUC = {round(AUC, 5)}')
        
        plt.title(f'ROC curve for all modalities for descriptor {descriptors[i]}')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc = 'lower right')
        i += 1

    plt.subplots_adjust(wspace = 0.15, hspace = 0.05)
    plt.show()
    

def seperate_dim_lit(image):
    
    ''' 0 => lit, 1 => lit because of open door, 2 => dim, 3 => dim with open door'''
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    average_brightness = gray_image.sum() / (gray_image.shape[0] * gray_image.shape[1])
    hist = np.histogram(gray_image, bins = np.arange(0, 255))[0]
    hist = hist / (hist.max() + 1e-7)
    hist_average_125_175 = hist[125:175].sum() / 50
        
    if average_brightness > 100:
        if hist_average_125_175 > 0.15:
            return 1
        else:
            return 0
    else:
        if hist_average_125_175 > 0.15:
            return 3
        else:
            return 2
    
def extract_face_info(img, mesh_detector):
    
    pts = face_mesh_mp(img, mesh_detector)
    
    if(pts is not None):
        nose_tip = pts[1]
        right_eye = pts[386]
        left_eye = pts[159]
        mouth = pts[14]
        
        nose_tip_depth = nose_tip[2]
        right_eye_depth = right_eye[2]
        left_eye_depth = left_eye[2]
        mouth_depth = mouth[2]
        
        right_eye_visible = True
        left_eye_visible = True
        mouth_visible = True
        
        difference_thresh = 0.125
        if right_eye_depth > (left_eye_depth + difference_thresh):
            right_eye_visible = False
        elif left_eye_depth > (right_eye_depth + difference_thresh):
            left_eye_visible = False
        if mouth_depth > (left_eye_depth + difference_thresh) or mouth_depth > (right_eye_depth + difference_thresh):
            mouth_visible = False
            
        return [right_eye_visible, left_eye_visible, mouth_visible]
    else:
        return [None, None, None]
        
        
        
def face_mesh_mp(img, mesh_detector):
    h, w = img.shape[:2]

    results = mesh_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    points = []
    
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            for landmark in face.landmark:
                points.append([int(landmark.x * w), int(landmark.y * h), landmark.z])
            break
        return points
    else:
        return None

def extract_faces_from_database(images_path, output_path, face_detection_confidence = 0.7):
    # mesh_detector = mp.solutions.face_mesh.FaceMesh(static_image_mode = True,
    #                                                 max_num_faces = 1,
    #                                                 refine_landmarks = True,
    #                                                 min_detection_confidence = face_detection_confidence)
    image_paths = []
    for dir, dirnames, filenames in os.walk(images_path):
        if len(filenames) > 0:
            for filename in filenames:
                if filename.endswith('.jpg'):
                    image_paths.append(os.path.join(dir, filename))
    
    detection_data = []
    # facenet = FaceNet()

    for i in tqdm(range(len(image_paths))):
        try:
            img = cv2.imread(image_paths[i])
            # lighting_condition = seperate_dim_lit(img)
            
            boxes = face_recognition.face_locations(img, 1, 'cnn')
            if len(boxes) <= 0:
                continue
            print('Found face!')
            # faces = [img[top : bottom, left : right] for top, right, bottom, left in boxes]
            # encodings = facenet.embeddings(faces)
            encodings = face_recognition.face_encodings(img, boxes, num_jitters = 3, model = 'large')
            data = [{'encoding' : encoding, 'location' : box, 'image path' : image_paths[i] } for encoding, box in zip(encodings, boxes)]
            detection_data.extend(data)
        except:
            continue
        
    encodings = [data['encoding'] for data in detection_data]
    print('Found this many encodings:', len(encodings))
    clt = SpectralClustering(n_clusters = 20, n_jobs = -1)
    clt.fit(encodings)
    
    padding = 3
    labelIDs = np.unique(clt.labels_)
    print(labelIDs)
    for labelID in labelIDs:
        indexes = np.where(clt.labels_ == labelID)[0]
        print('hi')
        if not os.path.isdir(os.path.join(output_path, str(labelID))):
            os.mkdir(os.path.join(output_path, str(labelID)))
        
        faces = []
        print(indexes)
        for i in indexes:
            face = cv2.imread(detection_data[i]['image path'])
            h,w = face.shape[:2]
            face_location = detection_data[i]['location']
            (top, right, bottom, left) = face_location
            
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(h, bottom + padding)
            right = min(w, right + padding)
            
            try:
                face = face[top : bottom, left : right]
                faces.append(cv2.resize(face, (128, 128)))
                # [right_eye_visible, left_eye_visible, mouth_visible] = extract_face_info(face, mesh_detector)
                
                # if right_eye_visible == None or left_eye_visible == None or mouth_visible == None:
                #     right_eye_visible = -1
                #     left_eye_visible = -1
                #     mouth_visible = -1
                
                # name = f'{i}_C_{str(int(lighting_condition))}_R_{str(int(right_eye_visible))}_L_{str(int(left_eye_visible))}_N_1_M_{str(int(mouth_visible))}.jpg'
                name = f'image {time.time()}.jpg'
                print(name)
                cv2.imwrite(os.path.join(output_path, str(labelID), name), face)
            except:
                continue
        
        

    