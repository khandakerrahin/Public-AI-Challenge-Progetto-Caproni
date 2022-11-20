#!/usr/bin/env python3
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog
import customtkinter
from os.path import expanduser

from classification import *
from metadata_extraction import *

home_dir = expanduser("~")

root = customtkinter.CTk()
root.title("model demo")
root.geometry('1024x768')
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")


# start frame
def start_frame(root, f=None):

    if f is not None:
        f.pack_forget()

    frame = customtkinter.CTkFrame(root)
    frame.pack(fill='both', expand=True)

    picture = tk.PhotoImage(file=os.path.join(".", "imgs", "airplane.png"))
    label = customtkinter.CTkLabel(frame, text='Welcome!', text_font=('Ubuntu Mono', 30),
                                   image=picture, compound='bottom')
    label.image = picture
    
    label.pack(ipadx=10, ipady=50)

    start_button = customtkinter.CTkButton(frame, text='Press here to start',  height=50, width=250,
                                           border_width=0, corner_radius=10,
                                           command=lambda: choose_model(frame))
    start_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    info_button = customtkinter.CTkButton(frame, text='Info',  height=50, width=250,
                                          border_width=0, corner_radius=10,
                                          command=lambda: get_info(frame))
    info_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

    hidden_frame = customtkinter.CTkFrame(frame, width=1, height=1)
    hidden_frame.place(in_=frame, anchor="n", relx=1, rely=0)
    eg_button = customtkinter.CTkButton(frame, #fg_color="#343A40", bg_color="#343A40", hover_color="#343A40",
                                        text='', height=10, width=10, border_width=0, corner_radius=0,
                                        command=lambda: popup_hippo(frame))
    eg_button.place(relx=0.55, rely=0.9, anchor=tk.W)


def new_frame(f):
    f.pack_forget()
    f = customtkinter.CTkFrame(root)
    f.pack(fill='both', expand=True)
    return f


# select existing or new model
def choose_model(f):
    f = new_frame(f)

    label = customtkinter.CTkLabel(f, text='Choose your task', text_font=("Ubuntu Mono", 18, 'bold'))
    
    label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    start_classification = customtkinter.CTkButton(
        f, text='Thematic Subdivision', height=100, width=300,
        border_width=0, corner_radius=10, fg_color="#7B2CBF", hover_color="#3C096C",
        command=lambda: classification(f, task="thematic_subdivision"))
    start_classification.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    start_metadata = customtkinter.CTkButton(
        f, text='Metadata Extraction',  height=100, width=300,
        border_width=0, corner_radius=10, fg_color="#7B2CBF", hover_color="#3C096C",
        command=lambda: metadata(f))
    start_metadata.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    start_damage = customtkinter.CTkButton(
        f, text='Damage Assessment', height=100, width=300,
        border_width=0, corner_radius=10, fg_color="#7B2CBF", hover_color="#3C096C",
        command=lambda: classification(f, task="damage_assessment"))
    start_damage.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    menu_button = customtkinter.CTkButton(f, text='Back',  height=50, width=250,
                                          border_width=0, corner_radius=10,
                                          command=lambda: start_frame(root, f))
    menu_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def classification(f, task):
    f = new_frame(f)

    folder = filedialog.askdirectory(initialdir=home_dir, title="Choose the image folder")

    label1 = customtkinter.CTkLabel(f, text=f'Image folder selected: {folder}',
                                    text_font=('Ubuntu Mono', 18))
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    b = customtkinter.CTkButton(f, text='Start',  height=50, width=250,
                                border_width=0, corner_radius=10,
                                command=lambda: start_task(f, folder_to_classify=folder, task=task))
    b.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    back_button = customtkinter.CTkButton(f, text='Back',  height=50, width=250,
                                          border_width=0, corner_radius=10,
                                          command=lambda: choose_model(f))
    back_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def metadata(f):
    f = new_frame(f)

    input_folder = filedialog.askdirectory(initialdir=home_dir, title="Choose the image folder")
    output_folder = filedialog.askdirectory(initialdir=home_dir, title="Choose the folder for the results")

    label1 = customtkinter.CTkLabel(f, text=f'Image folder selected: {input_folder}',
                                    text_font=('Ubuntu Mono', 18))
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label2 = customtkinter.CTkLabel(f,
                      text=f'Output folder selected: {output_folder}',
                      text_font=('Ubuntu Mono', 18))
    label2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    b = customtkinter.CTkButton(f, text='Start',  height=50, width=250,
                                border_width=0, corner_radius=10,
                                command=lambda: start_task(f, folder_to_classify=input_folder,
                                             output_folder=output_folder, task="metadata_extraction"))
    b.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    back_button = customtkinter.CTkButton(f, text='Back',  height=50, width=250,
                                          border_width=0, corner_radius=10,
                                          command=lambda: choose_model(f))
    back_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def start_task(f, folder_to_classify, output_folder=None, task="thematic_subdivision"):
    if task == "thematic_subdivision" or task == "damage_assessment":
        model = Classify(input_folder=folder_to_classify, task=task)
    elif task == "metadata_extraction":
        model = MetadataExtraction(folder_to_classify, output_folder)
    pb = threading.Thread(target=start_progress_bar,
                          args=(f, f'Starting {" ".join(task.split("_")).capitalize()}', model, task))
    pb.start()


def task_done(f, task):
    f = new_frame(f)
    label1 = customtkinter.CTkLabel(f, text=f"{' '.join(task.split('_')).capitalize()} done!",
                                    text_font=("Ubuntu Mono", 25, 'bold'))
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label2 = customtkinter.CTkLabel(f, text='press Menu to go on the main page \n '
                                            'press Close to exit.               ',
                                    text_font=('Ubuntu Mono', 18))
    label2.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    b1 = customtkinter.CTkButton(f, text='Menu',  height=50, width=250,
                                 border_width=0, corner_radius=10,
                                 command=lambda: start_frame(root, f))
    b1.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    b2 = customtkinter.CTkButton(f, text="Close",  height=50, width=250,
                                 border_width=0, corner_radius=10,
                                 command=lambda: root.quit())
    b2.place(relx=0.5, rely=0.8, anchor=tk.CENTER)


def popup(f, text):
    new_f = tk.Frame(f, width=300, height=250)
    new_f.config(bg='gray')
    new_f.place(in_=f, anchor="c", relx=.5, rely=.5)

    l = customtkinter.CTkLabel(new_f, text=text, text_font=('Ubuntu Mono', 18))
    l.config(bg='gray')
    l.pack(in_=new_f, anchor='w')

    close_b = customtkinter.CTkButton(new_f, text='Close',  height=50, width=250,
                                      border_width=0, corner_radius=10,
                                      command=lambda: new_f.place_forget())
    close_b.pack()


def popup_hippo(f):
    f = new_frame(f)

    picture = tk.PhotoImage(file=os.path.join(".", "imgs", "hippo.png")).subsample(3, 3)
    label = customtkinter.CTkLabel(f, text='Welcome!', text_font=('Ubuntu Mono', 30, 'bold'),
                                   image=picture, compound='bottom')
    label.image = picture
    
    label.pack(ipadx=10, ipady=50)

    start_button = customtkinter.CTkButton(f, text='Press here to start',   height=50, width=250,
                                           border_width=0, corner_radius=10,
                                           command=lambda: choose_model(f))
    start_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    info_button = customtkinter.CTkButton(f, text='Info',  height=50, width=250,
                                          border_width=0, corner_radius=10,
                                          command=lambda: get_info(f))
    info_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def get_info(f):
    f = new_frame(f)

    info_title = customtkinter.CTkLabel(f, text='Info',  text_font=('Ubuntu Mono', 25, 'bold'))
    info_title.place(relx=0.5, rely=0.05, anchor=tk.CENTER)

    # thematic subdivision
    label1 = customtkinter.CTkLabel(
        f, text='Thematic Subdivision                                                               '
                '                  ',
        text_font=('Ubuntu Mono', 13, 'bold'))
    label1.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

    label2 = customtkinter.CTkLabel(
        f, text="Select the folder where the images are saved. At the end of the process, a new folder "
                "'thematic_subdivision_result',\n "
                "inside the folder of unlabeled images will be created. It will contain the folders of "
                "classified images.             \n\n"
                "N.B images inside the new folder are a copy of the unlabeled images.                  "
                "                             ",
        anchor='w', text_font=("Ubuntu Mono", 12))
    label2.place(relx=0.5, rely=0.22, anchor=tk.CENTER)

    # metadata extraction
    label3 = customtkinter.CTkLabel(
        f, text='Metadata Extraction                                                                '
                 '                  ',
        text_font=("Ubuntu Mono", 13, " bold"))
    label3.place(relx=0.5, rely=0.35, anchor=tk.CENTER)

    label4 = customtkinter.CTkLabel(
        f, text='Select the folder where the images are saved and an output folder where you want to'
                ' save the results.               \n'
                "At the end of the process, a new CSV file 'metadata_results.csv', will be created in"
                " the provided                   \n"
                "folder. It will contain the image paths, their subject, the content, and the damage "
                "level.                          ",
        anchor='w', text_font=("Ubuntu Mono", 12))
    label4.place(relx=0.5, rely=0.42, anchor=tk.CENTER)

    # damage assessment
    label5 = customtkinter.CTkLabel(
        f, text='Damage Assessment                                                                  '
                '                  ',
        text_font=('Ubuntu Mono', 13, 'bold'))
    label5.place(relx=0.5, rely=0.55, anchor=tk.CENTER)

    label6 = customtkinter.CTkLabel(
        f, text="Select the folder where the images are saved. At the end of the process, a new folder "
                "'damage_assessment_result',   \n"
                "inside the folder of unlabeled images will be created. It will contain the folders of "
                "classified images.            \n\n"
                "N.B images inside the new folder are a copy of the unlabeled images.                  "
                "                             ",
        anchor='w', text_font=("Ubuntu Mono", 12))
    label6.place(relx=0.5, rely=0.62, anchor=tk.CENTER)

    # label5 = customtkinter.CTkLabel(
    #     f, text="Developed by Ludovico Maria Valenti, Alessia Meloni, Shaker Mahmud Khandaker, "
    #             "Md Ashraful Alam Hazari", text_font=('Ubuntu Mono', 9))
    # label5.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    menu_button = customtkinter.CTkButton(f, text='Back',  height=50, width=250,
                                          border_width=0, corner_radius=10,
                                          command=lambda: start_frame(root, f))
    menu_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def start_progress_bar(f, text, function, task='thematic_subdivision', text2=None):
    f = new_frame(f)

    l = customtkinter.CTkLabel(f, text=text, text_font=("Ubuntu Mono", 20, 'bold'))
    l.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    if text2:
        l1 = customtkinter.CTkLabel(f, text=text2, text_font=("Ubuntu Mono", 18, 'bold'))
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

