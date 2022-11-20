#!/usr/bin/env python3
import threading
import tkinter as tk
from tkinter import ttk, filedialog

from classification import *
from metadata_extraction import *


root = tk.Tk()
root.title("model demo")
root.geometry('1024x768')
root.config(bg='white')


# start frame
def start_frame(root, f=None):

    if f is not None:
        f.pack_forget()

    frame = tk.Frame(root)
    frame.config(bg='white')
    frame.pack(fill='both', expand=True)

    picture = tk.PhotoImage(file='./imgs/airplane.png')
    label = tk.Label(frame, text='Welcome!', font=('Ubuntu Mono', 18, 'bold'), image=picture, compound='bottom')
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
    hidden_frame.place(in_=frame, anchor="n", relx=0, rely=0)
    eg_button = tk.Button(hidden_frame, command=lambda: popup_hippo(frame))
    eg_button.config(bg='white', fg='white', highlightthickness=0, borderwidth=-1)
    eg_button.pack()


def new_frame(f):
    f.pack_forget()
    f = tk.Frame(root)
    f.config(bg='white')
    f.pack(fill='both', expand=True)
    return f


# select existing or new model
def choose_model(f):
    f = new_frame(f)

    label = tk.Label(f, text='Choose your task', font=("Ubuntu Mono", 18, 'bold'))
    label.config(bg='white')
    label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    start_classification = tk.Button(f, text='Thematic Subdivision', height=4, width=30,
                                     command=lambda: classification(f, task="thematic_subdivision"))
    start_classification.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    start_metadata = tk.Button(f, text='Metadata Extraction', height=4, width=30, command=lambda: metadata(f))
    start_metadata.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    start_damage = tk.Button(f, text='Damage Assessment', height=4, width=30,
                             command=lambda: classification(f, task="damage_assessment"))
    start_damage.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    menu_button = tk.Button(f, text='Back', height=2, width=10, command=lambda: start_frame(root, f))
    menu_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def classification(f, task):
    f = new_frame(f)

    folder = filedialog.askdirectory(initialdir="/", title="Choose the image folder")

    label1 = tk.Label(f,
                      text=f'Image folder selected: {folder}',
                      font=('Ubuntu Mono', 12))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    b = tk.Button(f, text='Start', height=2, width=30,
                  command=lambda: start_task(f, folder_to_classify=folder, task=task))
    b.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    back_button = tk.Button(f, text='Back', height=2, width=10, command=lambda: choose_model(f))
    back_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def metadata(f):
    f = new_frame(f)

    input_folder = filedialog.askdirectory(initialdir="/", title="Choose the image folder")
    output_folder = filedialog.askdirectory(initialdir="/", title="Choose the folder for the results")

    label1 = tk.Label(f,
                      text=f'Image folder selected: {input_folder}',
                      font=('Ubuntu Mono', 15))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label2 = tk.Label(f,
                      text=f'Output folder selected: {output_folder}',
                      font=('Ubuntu Mono', 15))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    b = tk.Button(f, text='Start', height=2, width=30,
                  command=lambda: start_task(f, folder_to_classify=input_folder,
                                             output_folder=output_folder, task="metadata_extraction"))
    b.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    back_button = tk.Button(f, text='Back', height=2, width=10, command=lambda: choose_model(f))
    back_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def start_task(f, folder_to_classify, output_folder=None, task="thematic_subdivision"):
    if task == "thematic_subdivision" or task == "damage_assessment":
        model = Classify(input_folder=folder_to_classify, task=task)
    elif task == "metadata_extraction":
        model = MetadataExtraction(folder_to_classify, output_folder)
    pb = threading.Thread(target=start_progress_bar, args=(f, f'Starting {" ".join(task.split("_")).capitalize()}', model, task))
    pb.start()


def task_done(f, task):
    f = new_frame(f)
    label1 = tk.Label(f, text=f"{' '.join(task.split('_')).capitalize()} done!", font=("Ubuntu Mono", 18, 'bold'))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label2 = tk.Label(f, text='press Menu to go on the main page \n '
                              'press Close to exit.               ',
                      font=('Ubuntu Mono', 16))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    b1 = tk.Button(f, text='Menu', height=2, width=30, command=lambda: start_frame(root, f))
    b1.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    b2 = tk.Button(f, text="Close", height=2, width=30, command=lambda: root.quit())
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
    f = new_frame(f)

    picture = tk.PhotoImage(file='./imgs/hippo.png').subsample(5, 5)
    label = tk.Label(f, text='Welcome!', font=('Ubuntu Mono', 18, 'bold'), image=picture, compound='bottom')
    label.image = picture
    label.config(bg='white')
    label.pack(ipadx=10, ipady=50)

    start_button = tk.Button(f, text='Press here to start',  height=2, width=15,
                             command=lambda: choose_model(f))
    start_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    info_button = tk.Button(f, text='Info', height=2, width=15, command=lambda: get_info(f))
    info_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def get_info(f):
    f = new_frame(f)

    info_title = tk.Label(f, text='Info',  font=('Ubuntu Mono', 18, 'bold'))
    info_title.config(bg='white')
    info_title.place(relx=0.5, rely=0.05, anchor=tk.CENTER)

    # thematic subdivision
    label1 = tk.Label(f, text='Thematic Subdivision                                                               '
                              '                  ',
                      font=('Ubuntu Mono', 13, 'bold'))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

    label2 = tk.Label(f,
                      text="Select the folder where the images are saved. At the end of the process, a new folder "
                           "'thematic_subdivision_result',\n "
                           "inside the folder of unlabeled images will be created. It will contain the folders of "
                           "classified images.             \n\n"
                           "N.B images inside the new folder are a copy of the unlabeled images.                  "
                           "                             ",
                      anchor='w', font=("Ubuntu Mono", 12))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.22, anchor=tk.CENTER)

    # metadata extraction
    label3 = tk.Label(f, text='Metadata Extraction                                                                '
                              '                  ',
                      font=("Ubuntu Mono", 13, " bold"))
    label3.config(bg='white')
    label3.place(relx=0.5, rely=0.35, anchor=tk.CENTER)
    label4 = tk.Label(f, text='Select the folder where the images are saved and an output folder where you want to'
                              ' save the results.               \n'
                              "At the end of the process, a new CSV file 'metadata_results.csv', will be created in"
                              " the provided                   \n"
                              "folder. It will contain the image paths, their subject, the content, and the damage "
                              "level.                          ",
                      anchor='w', font=("Ubuntu Mono", 12))
    label4.config(bg='white')
    label4.place(relx=0.5, rely=0.42, anchor=tk.CENTER)

    # damage assessment
    label5 = tk.Label(f, text='Damage Assessment                                                                  '
                              '                  ',
                      font=('Ubuntu Mono', 13, 'bold'))
    label5.config(bg='white')
    label5.place(relx=0.5, rely=0.55, anchor=tk.CENTER)

    label6 = tk.Label(f, text="Select the folder where the images are saved. At the end of the process, a new folder "
                              "'damage_assessment_result',   \n"
                              "inside the folder of unlabeled images will be created. It will contain the folders of "
                              "classified images.            \n\n"
                              "N.B images inside the new folder are a copy of the unlabeled images.                  "
                              "                             ",
                      anchor='w', font=("Ubuntu Mono", 12))
    label6.config(bg='white')
    label6.place(relx=0.5, rely=0.62, anchor=tk.CENTER)

    #
    # label5 = tk.Label(f, text="Developed by Ludovico Maria Valenti, Alessia Meloni, Shaker Mahmud Khandaker, "
    #                           "Md Ashraful Alam Hazari", font=('Ubuntu Mono', 9))
    # label5.config(bg='white')
    # label5.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    menu_button = tk.Button(f, text='Back', height=2, width=10, command=lambda: start_frame(root, f))
    menu_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


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
                           "the path of the images, the subject, the content, the description, and the damege  "
                           "level.                                                                             \n",
                      font=('Ubuntu Mono', 12), anchor='w')
        l1.config(bg='gray')
        l2.config(bg='gray')
        l1.pack()
        l2.pack()

    close_b = tk.Button(new_f, text='Close', command=lambda: new_f.place_forget())
    close_b.config(bg='white')
    close_b.pack()


def start_progress_bar(f, text, function, task='thematic_subdivision', text2=None):
    f = new_frame(f)

    l = tk.Label(f, text=text, font=("Ubuntu Mono", 20, 'bold'))
    l.config(bg='white')
    l.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    if text2:
        l1 = tk.Label(f, text=text2, font=("Ubuntu Mono", 18, 'bold'))
        l1.config(bg='white')
        l1.place(relx=0.5, rely=0.45, anchor=tk.CENTER)

    pb = ttk.Progressbar(
        f,
        orient='horizontal',
        mode='indeterminate',
        length=450
    )
    # place the progressbar
    pb.place(relx=0.5, rely=0.55, anchor=tk.CENTER)

    pb.start()

    if task == 'thematic_subdivision' or task == 'damage_assessment':
        t1 = threading.Thread(target=function.classify)
        t1.start()
        t1.join()
        pb.stop()
        task_done(f, task)

    elif task == 'metadata_extraction':
        # t1 = threading.Thread(target=function.get_metadata)
        start_progress_bar(f, "Working on metadata extraction", function, task='s',
                           text2="1. Subject extraction              ")

    elif task == "s":
        t1 = threading.Thread(target=function.get_subject)
        t1.start()
        t1.join()
        pb.stop()
        start_progress_bar(f, "Working on metadata extraction", function, task='c',
                           text2="2. Content extraction              ")
    elif task == "c":
        t1 = threading.Thread(target=function.get_content)
        t1.start()
        t1.join()
        pb.stop()
        start_progress_bar(f, "Working on metadata extraction", function, task='d',
                           text2="3. Description                     ")
    elif task == "d":
        t1 = threading.Thread(target=function.get_description)
        t1.start()
        t1.join()
        pb.stop()
        start_progress_bar(f, "Working on metadata extraction", function, task='dmg',
                           text2="4. Damage assessment               ")
    elif task == 'dmg':
        t1 = threading.Thread(target=function.get_damage)
        t1.start()
        t1.join()
        pb.stop()
        function.get_metadata()
        task_done(f, "metadata_extraction")

    else:
        t1 = None
        raise ValueError("task not implemented")


start_frame(root)

root.mainloop()

