#!/usr/bin/env python3
import threading
import tkinter as tk
from tkinter import ttk

from final_model import *

root = tk.Tk()
root.title("model demo")
root.geometry('700x600')
root.config(bg='white')


# start frame
def start_frame(root, f=None):

    if f is not None:
        f.pack_forget()

    frame = tk.Frame(root, width=700, height=600)
    frame.config(bg='white')
    frame.pack(fill='both', expand=True)

    picture = tk.PhotoImage(file='./imgs/airplane.png')
    label = tk.Label(frame, text='Welcome!', font=('Ubuntu Mono', 18), image=picture, compound='bottom')
    label.image = picture
    label.config(bg='white')
    label.pack(ipadx=10, ipady=50)

    start_button = tk.Button(frame, text='Press here to start',  height=2, width=15,
                             command=lambda: choose_model(frame))
    start_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    info_button = tk.Button(frame, text='Info', height=2, width=15, command=lambda: get_info(frame))
    info_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

    hidden_frame = tk.Frame(frame, width=1, height=1)
    hidden_frame.config(bg='white', highlightthickness=0, borderwidth=-1)
    hidden_frame.place(in_=frame, anchor="n", relx=.55, rely=0.44)
    eg_button = tk.Button(hidden_frame, command=lambda: popup_hippo(frame))
    eg_button.config(bg='white', fg='white', highlightthickness=0, borderwidth=-1)
    eg_button.pack()


# select existing or new model
def choose_model(f):
    f.pack_forget()
    f = tk.Frame(root, width=700, height=600)
    f.config(bg='white')
    f.pack(fill='both', expand=True)

    start_existing_model = tk.Button(f, text='Existing model', height=5, width=30,
                                     command=lambda: existing_model(f))
    start_existing_model.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    start_new_model = tk.Button(f, text='New model',  height=5, width=30,
                                command=lambda: new_model(f))
    start_new_model.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    menu_button = tk.Button(f, text='Back', height=2, width=10, command=lambda: start_frame(root, f))
    menu_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)


def existing_model(f):
    f.pack_forget()

    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    label1 = tk.Label(new_f,
                      text='Please, provide the path of the model',
                      font=('Ubuntu Mono', 12))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    input_text1 = tk.Text(new_f, height=2, width=60)
    input_text1.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    label2 = tk.Label(new_f,
                      text='Please, provide the path of the folder with images to classify',
                      font=('Ubuntu Mono', 12))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    input_text2 = tk.Text(new_f, height=2, width=60)
    input_text2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    b = tk.Button(new_f, text='Start', height=2, width=30,
                  command=lambda: load_model(new_f, input_text2, input_text1))
    b.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    back_button = tk.Button(new_f, text='Back', height=2, width=10, command=lambda: choose_model(new_f))
    back_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def load_model(f, text1, text2):
    folder_to_classify = text1.get(1.0, "end-1c")
    model_folder = text2.get(1.0, "end-1c")
    model_exists = True
    folder_exists = True

    if not os.path.exists(folder_to_classify):
        popup(f, "The path for label inference does not exist!")
        folder_exists = False
    if not os.path.exists(model_folder):
        popup(f, "The path for the model does not exist!")
        model_exists = False
    if os.path.exists(model_folder) and len(os.listdir(model_folder)) == 0:
        popup(f, "The model folder is empty!")
        model_exists = False

    if model_exists and folder_exists:
        model = Train(input_folder=None, output_folder=folder_to_classify, model_folder=model_folder)

        start_classification(f, model)


# new model frame
def new_model(f):
    f.pack_forget()

    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    label1 = tk.Label(new_f,
                      text='Please, provide the path to labeled folders',
                      font=('Ubuntu Mono', 12))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    input_text1 = tk.Text(new_f, height=2, width=60)
    input_text1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label2 = tk.Label(new_f,
                      text='Please, provide the path of the folder with images to classify',
                      font=('Ubuntu Mono', 12))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    input_text2 = tk.Text(new_f, height=2, width=60)
    input_text2.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    label3 = tk.Label(new_f,
                      text='Please, provide the path to save the model (existing or new)',
                      font=('Ubuntu Mono', 12))
    label3.config(bg='white')
    label3.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    input_text3 = tk.Text(new_f, height=2, width=60)
    input_text3.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    b = tk.Button(new_f, text='Start', height=2, width=30,
                  command=lambda: create_model(new_f, input_text1, input_text2, input_text3))
    b.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    back_button = tk.Button(new_f, text='Back', height=2, width=10, command=lambda: choose_model(new_f))
    back_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def create_model(f, text1, text2, text3):
    labeled_folders = text1.get(1.0, "end-1c")
    folder_to_classify = text2.get(1.0, "end-1c")
    model_folder = text3.get(1.0, "end-1c")

    labeled_exists = True
    classification_exists = True

    if not os.path.exists(labeled_folders):
        popup(f, "You must specify an existing path for labeled folders")
        labeled_exists = False

    if not os.path.exists(folder_to_classify):
        popup(f, "The path for label inference does not exist!")
        classification_exists = False

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    if labeled_exists and classification_exists:
        trainer = Train(labeled_folders, folder_to_classify, model_folder)
        start_training(f, trainer)


def start_training(f, trainer):
    pb = threading.Thread(target=start_progress_bar, args=(f, 'Starting training', trainer, 'T'))
    pb.start()


def training_done(f, trainer):
    f.pack_forget()
    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    label2 = tk.Label(new_f, text='Training done!', font=('Ubuntu Mono', 18))
    label2.config(bg='white')
    label2.pack(ipadx=10, ipady=30)

    label3 = tk.Label(f, text=f'Model saved in {trainer.output_folder}', font=("Ubuntu Mono", 18))
    label3.pack(ipadx=10, ipady=30)
    stop_or_continue(new_f, trainer)


def stop_or_continue(f, trainer):
    f.pack_forget()
    f = tk.Frame(root, width=700, height=600)
    f.config(bg='white')
    f.pack(fill='both', expand=True)

    label1 = tk.Label(f,
                      text='Training done',
                      font=('Ubuntu Mono', 18))
    label1.config(bg='white')
    label1.pack(ipadx=10, ipady=30)

    classification = tk.Button(f, text='Start classification', command=lambda: start_classification(f, trainer))
    classification.pack(ipadx=10, ipady=10, expand=True)

    stop_process = tk.Button(f, text='Stop process', command=lambda: root.quit())
    stop_process.pack(ipadx=10, ipady=10, expand=True)


def start_classification(f, trainer):
    pb = threading.Thread(target=start_progress_bar, args=(f, 'Starting Classification', trainer, 'C'))
    pb.start()


def classification_done(f):
    f.pack_forget()
    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    label2 = tk.Label(new_f, text='Classification done! \n press the button to close', font=('Ubuntu Mono', 18))
    label2.config(bg='white')
    label2.pack(ipadx=10, ipady=30)

    b = tk.Button(new_f, text='Close', height=1, width=30,
                  command=lambda: root.quit())
    b.pack(ipadx=10, ipady=10, expand=True)


def popup(f, text):
    new_f = tk.Frame(f, width=300, height=250)
    new_f.config(bg='gray')
    new_f.place(in_=f, anchor="c", relx=.5, rely=.5)

    l = tk.Label(new_f, text=text, font=('Ubuntu Mono', 12))
    l.config(bg='gray')
    l.pack( in_=new_f, anchor='w')

    close_b = tk.Button(new_f, text='Close', command=lambda: new_f.place_forget())
    close_b.config(bg='white')
    close_b.pack()


def popup_hippo(f):
    f.pack_forget()
    frame = tk.Frame(root, width=700, height=600)
    frame.config(bg='white')
    frame.pack(fill='both', expand=True)

    picture = tk.PhotoImage(file='./imgs/hippo.png').subsample(5, 5)
    label = tk.Label(frame, text='Welcome!', font=('Ubuntu Mono', 18), image=picture, compound='bottom')
    label.image = picture
    label.config(bg='white')
    label.pack(ipadx=10, ipady=50)

    start_button = tk.Button(frame, text='Press here to start', command=lambda: choose_model(frame))
    start_button.pack(ipadx=5, ipady=5, expand=True)


def get_info(f):
    f.pack_forget()
    f = tk.Frame(root, width=700, height=600)
    f.config(bg='white')
    f.pack(fill='both', expand=True)

    info_title = tk.Label(f, text='Info',  font=('Ubuntu Mono', 18))
    info_title.config(bg='white')
    info_title.pack()

    train_and_classification = tk.Button(f, text='Train & Classification',
                                         command=lambda: info_popup(f, 'tc'))
    train_and_classification.pack(ipadx=5, ipady=5, expand=True)

    classification = tk.Button(f, text='Classification',
                               command=lambda: info_popup(f, 'c'))
    classification.pack(ipadx=5, ipady=5, expand=True)

    menu_button = tk.Button(f, text='Back', command=lambda: start_frame(root, f))
    menu_button.pack(ipadx=5, ipady=5, expand=True)


def info_popup(f, text):
    new_f = tk.Frame(f, width=300, height=250)
    new_f.config(bg='gray')
    new_f.place(in_=f, anchor="c", relx=.5, rely=.5)

    if text == 'c':
        l1 = tk.Label(new_f,
                      text="1. Provide the path for: unlabeled images, folder where the model is saved.\n",
                      font=('Ubuntu Mono', 12), anchor='w')
        l2 = tk.Label(new_f,
                      text="At the end of the process, a new folder 'classification_result' inside the \n"
                           "folder of unlabeled images will be created. It will contain the folders of \n"
                           "classified images.                                                         \n",
                      font=('Ubuntu Mono', 12), anchor='w')
        l1.config(bg='gray')
        l2.config(bg='gray')
        l1.pack()
        l2.pack()
    else:
        l1 = tk.Label(new_f,
                      text="1. Create a folder with labeled images: it must contain a folder for each class.",
                      font=('Ubuntu Mono', 12), anchor='w')
        l2 = tk.Label(new_f,
                      text="    * suggested ~100 images per class                                              ",
                      font=('Ubuntu Mono', 12), anchor='w')
        l3 = tk.Label(new_f,
                      text="2. provide the path for: labeled images, unlabeled images, folder where the model\n"
                           "    will be saved.                                                                  \n\n"
                           "At the end of the process, a new folder 'classification_result' inside the folder of\n"
                           "unlabeled images will be created. It will contain the folders of classified images.\n",
                      font=('Ubuntu Mono', 12), anchor='w')
        l1.config(bg='gray')
        l2.config(bg='gray')
        l3.config(bg='gray')
        l1.pack()
        l2.pack()
        l3.pack()

    close_b = tk.Button(new_f, text='Close', command=lambda: new_f.place_forget())
    close_b.config(bg='white')
    close_b.pack()


def start_progress_bar(f, text, function, task):
    f.pack_forget()

    f = tk.Frame(root, width=700, height=600)
    f.config(bg='white')
    f.pack(fill='both', expand=True)

    l = tk.Label(f, text=text, font=("Ubuntu Mono", 14))
    l.config(bg='white')
    l.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    pb = ttk.Progressbar(
        f,
        orient='horizontal',
        mode='indeterminate',
        length=280
    )
    # place the progressbar
    pb.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    pb.start(5)

    if task == 'T':
        t1 = threading.Thread(target=function.train)
    else:
        t1 = threading.Thread(target=function.classify)

    t1.start()
    t1.join()
    pb.stop()

    if task == 'T':
        training_done(f, function)
    else:
        classification_done(f)


start_frame(root)

root.mainloop()

