import cv2, numpy as np, utilities, os, tkinter as tk, pickle, tensorflow as tf
from tkinter import messagebox, filedialog, simpledialog

class Program(object):
    def __init__(self, database_path = os.path.join('.', 'Database'), apply_clahe = True, threshold = 0.95):
        self.cap = cv2.VideoCapture(0)
        self.main_window_name = 'Output'
        self.add_subject_window_name = 'Adding a new subject'
        self.database_path = database_path
        self.font = ('Consolas', 18)
        self.width, self.height = 800, 600
        self.window = tk.Tk()
        self.window.title('Without classifier')
        self.window.maxsize(self.width, self.height)
        self.window.minsize(self.width, self.height)
        self.frames_count = 100
        self.threshold = threshold
        self.apply_clahe = apply_clahe
        
    def main(self):
        if self.apply_clahe:
            clahe = cv2.createCLAHE(3, (6, 6))
            
        if os.path.isfile('models/svm classifier/model.joblib'):
            with open('models/svm classifier/model.joblib', 'rb') as file:
                classifier = pickle.load(file)
        if os.path.isfile('models/svm classifier/info.joblib'):
            with open('models/svm classifier/info.joblib', 'rb') as info_file:
                subjects = pickle.load(info_file)
                
        feature_extractor = tf.lite.Interpreter(model_path = './models/facenet optimized/model.tflite')
        feature_extractor.allocate_tensors()
        feature_extractor_input_details = feature_extractor.get_input_details()
        feature_extractor_output_details = feature_extractor.get_output_details()
        
        
        
        cv2.namedWindow(self.main_window_name)
        while self.cap.isOpened() and cv2.getWindowProperty(self.main_window_name, cv2.WND_PROP_VISIBLE) == 1:
            ret, frame = self.cap.read()
            if self.apply_clahe:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                frame[:, :, 0] = clahe.apply(frame[:, :, 0])
                frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
                
            dets = utilities.detect_faces_mp(frame)
            faces = utilities.preprocess_faces(frame, dets)
            
            for det, face in zip(dets, faces):
                face = np.expand_dims(face, axis = 0)
                feature_extractor.set_tensor(feature_extractor_input_details[0]['index'], face.astype(np.float32))
                feature_extractor.invoke()
                feature = feature_extractor.get_tensor(feature_extractor_output_details[0]['index'])
                prediction = classifier.predict_proba(feature)[0]
                p = np.argmax(prediction)
                prediction = prediction[p]
                if prediction <= self.threshold:
                    name = 'Unknown'
                else:
                    name = subjects[p]
                    
                x, y, w, h = det
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(frame, name, (x, y-20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                
            cv2.imshow(self.main_window_name, frame)
            key = cv2.waitKey(1)
        cv2.destroyAllWindows()
        
    def add_subject(self):
        if self.apply_clahe:
            clahe = cv2.createCLAHE(3, (6, 6))
            
        subject_name = simpledialog.askstring('Subject name', 'Please enter the name of the subject you wish to add')
        if subject_name == None:
            messagebox.showerror('Invalid name!', 'Invalid subject name!')
            return
        
        output_dir = os.path.join(self.database_path, subject_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        else:
            messagebox.showinfo('Already exists', 'The subject already exists in the database.')
            return
            
        cv2.namedWindow(self.add_subject_window_name)
        i = 0
        while self.cap.isOpened() and cv2.getWindowProperty(self.add_subject_window_name, cv2.WND_PROP_VISIBLE) == 1 and i < self.frames_count:
            ret, frame = self.cap.read()
            if self.apply_clahe:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                frame[:, :, 0] = clahe.apply(frame[:, :, 0])
                frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
            dets = utilities.detect_faces_mp(frame)
            if len(dets) > 1:
                messagebox.showinfo('Too many subjects!', 'Only a single subject must be in the frame!')
                continue
            elif len(dets) == 1:
                x, y, w, h = dets[0]
                face = frame[y : y+h, x : x+w]
                cv2.imwrite(os.path.join(self.database_path, subject_name, f'{i}.jpg'), face)
                i += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            
            cv2.imshow(self.add_subject_window_name, frame)
            key = cv2.waitKey(100)
        cv2.destroyAllWindows()
    
    def train(self):
        messagebox.showinfo('Training in progress...', 'Close this message and check the terminal as the training is in progress!')
        output = utilities.train_svm()
        
        if output[1] == -1:
            messagebox.showerror('Error', f'Subject {output[0]} has less than 10 images!')
            return
        output_path = 'models/svm classifier'
        with open(os.path.join(output_path, 'model.joblib'), 'wb') as file:
            pickle.dump(output[0], file)
        
        with open(os.path.join(output_path, 'info.joblib'), 'wb') as info_file:
            pickle.dump(output[2], info_file)

    def run(self):
        new_subject_btn = tk.Button(master = self.window, text = 'Add new subject', command = self.add_subject, font = self.font)
        start_btn = tk.Button(master = self.window, text = 'start', command = self.main, font = self.font)
        train_btn = tk.Button(master = self.window, text = 'Train the model', command = self.train, font = self.font)
        
        new_subject_btn.place(x = (self.width/2 - new_subject_btn.winfo_reqwidth()/2), y = 50)
        train_btn.place(x = self.width/2 - train_btn.winfo_reqwidth()/2, y = self.height / 2)
        start_btn.place(x = (self.width/2 - start_btn.winfo_reqwidth()/2), y = self.height - 100)
        self.window.mainloop()

program = Program()
program.run()

        