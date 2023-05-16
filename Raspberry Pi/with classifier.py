import cv2, numpy as np, utilities, os, tkinter as tk
from tkinter import messagebox, filedialog, simpledialog

class Program(object):
    def __init__(self, database_path = os.path.join('.', 'Raspberry Pi', 'Database')):
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
        
    def main(self):
        cv2.namedWindow(self.main_window_name)
        while self.cap.isOpened() and cv2.getWindowProperty(self.main_window_name, cv2.WND_PROP_VISIBLE) == 1:
            ret, frame = self.cap.read()
            dets = utilities.detect_faces_mp(frame)
            for x, y, w, h in dets:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.imshow(self.main_window_name, frame)
            key = cv2.waitKey(1)
        cv2.destroyAllWindows()
        
    def add_subject(self):
        subject_name = simpledialog.askstring('Subject name', 'Please enter the name of the subject you wish to add')
        if subject_name == '':
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
            dets = utilities.detect_faces_mp(frame)
            if len(dets) > 1:
                messagebox.showinfo('Too many subjects!', 'Only a single subject must be in the frame!')
                continue
            elif len(dets) == 1:
                x, y, w, h = dets[0]
                face = frame[y:y + h, x : x + w]
                cv2.imwrite(os.path.join(output_dir, f'{i}.jpg'), face)
                i += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            
            cv2.imshow(self.add_subject_window_name, frame)
            key = cv2.waitKey(100)
        cv2.destroyAllWindows()
    


    def run(self):
        new_subject_btn = tk.Button(master = self.window, text = 'Add new subject', command = self.add_subject, font = self.font)
        start_btn = tk.Button(master = self.window, text = 'start', command = self.main, font = self.font)
        
        new_subject_btn.place(x = (self.width/2 - new_subject_btn.winfo_reqwidth()/2), y = 50)
        start_btn.place(x = (self.width/2 - start_btn.winfo_reqwidth()/2), y = self.height - 100)
        self.window.mainloop()

program = Program()
program.run()

        