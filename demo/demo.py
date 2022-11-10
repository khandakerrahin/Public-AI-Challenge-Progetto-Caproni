#!/usr/bin/env python3
import threading
import tkinter as tk
from tkinter import ttk, filedialog

from thematic_subdivision import *
from metadata_extraction import *

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

    # hidden_frame = tk.Frame(frame, width=1, height=1)
    # hidden_frame.config(bg='white', highlightthickness=0, borderwidth=-1)
    # hidden_frame.place(in_=frame, anchor="n", relx=.55, rely=0.44)
    # eg_button = tk.Button(hidden_frame, command=lambda: popup_hippo(frame))
    # eg_button.config(bg='white', fg='white', highlightthickness=0, borderwidth=-1)
    # eg_button.pack()


# select existing or new model
def choose_model(f):
    f.pack_forget()
    f = tk.Frame(root, width=700, height=600)
    f.config(bg='white')
    f.pack(fill='both', expand=True)

    start_classification = tk.Button(f, text='Classification', height=5, width=30,
                                     command=lambda: classification(f))
    start_classification.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    start_metadata = tk.Button(f, text='Metadata Extraction', height=5, width=30, command=lambda: metadata(f))
    start_metadata.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    start_segmentation = tk.Button(f, text='Damage Identification', height=5, width=30)
    start_segmentation.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    menu_button = tk.Button(f, text='Back', height=2, width=10, command=lambda: start_frame(root, f))
    menu_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)


def classification(f):
    f.pack_forget()

    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    input_text = filedialog.askdirectory(initialdir="/", title="Choose the image folder")

    label1 = tk.Label(new_f,
                      text=f'Image folder selected: {input_text}',
                      font=('Ubuntu Mono', 12))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    b = tk.Button(new_f, text='Start', height=2, width=30,
                  command=lambda: start_classification(new_f, input_text))
    b.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    back_button = tk.Button(new_f, text='Back', height=2, width=10, command=lambda: choose_model(new_f))
    back_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def start_classification(f, folder_to_classify):
    model = Classify(input_folder=folder_to_classify)

    pb = threading.Thread(target=start_progress_bar,
                      args=(f, 'Starting Classification', model, 'c'))
    pb.start()


def classification_done(f):
    f.pack_forget()
    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    label2 = tk.Label(new_f, text='Classification done! \n '
                                  'press Menu to go on the main page \n '
                                  'press Close to exit.',
                      font=('Ubuntu Mono', 18))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    b1 = tk.Button(new_f, text='Menu', height=2, width=30, command=lambda: start_frame(root, new_f))
    b1.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    b2 = tk.Button(new_f, text="Close", height=2, width=30, command=lambda: root.quit())
    b2.place(relx=0.5, rely=0.8, anchor=tk.CENTER)


def metadata(f):
    f.pack_forget()

    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    input_text1 = filedialog.askdirectory(initialdir="/", title="Choose the image folder")
    input_text2 = filedialog.askdirectory(initialdir="/", title="Choose the folder for the results")

    label1 = tk.Label(new_f,
                      text=f'Image folder selected: {input_text1}',
                      font=('Ubuntu Mono', 12))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label2 = tk.Label(new_f,
                      text=f'Output folder selected: {input_text2}',
                      font=('Ubuntu Mono', 12))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    b = tk.Button(new_f, text='Start', height=2, width=30,
                  command=lambda: start_metadata_extraction(new_f, input_text1, input_text2))
    b.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    back_button = tk.Button(new_f, text='Back', height=2, width=10, command=lambda: choose_model(new_f))
    back_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def start_metadata_extraction(f, image_folder, output_folder, task='s'):
    ME = MetadataExtraction(image_folder, output_folder)
    pb = threading.Thread(target=start_progress_bar,
                          args=(f, 'Starting Metadata Extraction', ME, 'me'))
    pb.start()


def metadata_extraction_done(f):
    f.pack_forget()
    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    label2 = tk.Label(new_f, text='Metadata Extraction done! \n\n '
                                  'press the menu button to go on the main page \n '
                                  'press Close to exit.',
                      font=('Ubuntu Mono', 18))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    b1 = tk.Button(new_f, text='Menu', height=2, width=30, command=lambda: start_frame(root, new_f))
    b1.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    b2 = tk.Button(new_f, text="Close", height=2, width=30, command=lambda: root.quit())
    b2.place(relx=0.5, rely=0.8, anchor=tk.CENTER)


def popup(f, text):
    new_f = tk.Frame(f, width=300, height=250)
    new_f.config(bg='gray')
    new_f.place(in_=f, anchor="c", relx=.5, rely=.5)

    l = tk.Label(new_f, text=text, font=('Ubuntu Mono', 12))
    l.config(bg='gray')
    l.pack(in_=new_f, anchor='w')

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

    classification = tk.Button(f, text='Classification',
                               command=lambda: info_popup(f, 'c'))
    classification.pack(ipadx=5, ipady=5, expand=True)

    metadata_extraction = tk.Button(f, text='Metadata Extraction',
                                         command=lambda: info_popup(f, 'tc'))
    metadata_extraction.pack(ipadx=5, ipady=5, expand=True)

    menu_button = tk.Button(f, text='Back', command=lambda: start_frame(root, f))
    menu_button.pack(ipadx=5, ipady=5, expand=True)


def info_popup(f, text):
    new_f = tk.Frame(f, width=300, height=250)
    new_f.config(bg='gray')
    new_f.place(in_=f, anchor="c", relx=.5, rely=.5)

    if text == 'c':
        l1 = tk.Label(new_f,
                      text="1. Provide the path for the images\n",
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
                      text="1. Provide the path for the images\n",
                      font=('Ubuntu Mono', 12), anchor='w')

        l2 = tk.Label(new_f,
                      text="At the end of the process, a CSV file will be created. It will contain:            \n"
                           "the path of the images, the subject, the content, and th description.              \n",
                      font=('Ubuntu Mono', 12), anchor='w')
        l1.config(bg='gray')
        l2.config(bg='gray')
        l1.pack()
        l2.pack()

    close_b = tk.Button(new_f, text='Close', command=lambda: new_f.place_forget())
    close_b.config(bg='white')
    close_b.pack()


def start_progress_bar(f, text, function, task='c'):
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

    pb.start()

    if task == 'c':
        t1 = threading.Thread(target=function.classify)

    elif task == 'me':
        t1 = threading.Thread(target=function.get_metadata)

    t1.start()
    t1.join()
    pb.stop()

    if task == 'c':
        classification_done(f)
    elif task == 'me':
        metadata_extraction_done(f)


start_frame(root)

root.mainloop()

