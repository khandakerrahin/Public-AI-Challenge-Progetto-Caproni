#!/usr/bin/env python3

import time
import tkinter as tk
from tkinter import ttk

import os
from final_model import *

root = tk.Tk()
root.title("model demo")
root.geometry('700x600')
root.config(bg='white')


# start frame
def start_frame(root):
    frame = tk.Frame(root, width=700, height=600)
    frame.config(bg='white')
    frame.pack(fill='both', expand=True)

    picture = tk.PhotoImage(file='airplane.png')
    label = tk.Label(frame, text='Welcome!', font=('Ubuntu Mono', 18), image=picture, compound='bottom')
    label.image = picture
    label.config(bg='white')
    label.pack(ipadx=10, ipady=50)

    start_button = tk.Button(frame, text='Press here to start', command=lambda: choose_model(frame))
    start_button.pack(ipadx=5, ipady=5, expand=True)

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


def existing_model(f):
    f.pack_forget()

    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    label1 = tk.Label(new_f,
                      text='Please, provide the path of the model',
                      font=('Ubuntu Mono', 12))
    label1.config(bg='white')
    label1.pack(ipadx=10, ipady=10, expand=True)

    input_text1 = tk.Text(new_f, height=2, width=60)
    input_text1.pack(ipadx=10, ipady=10, expand=False)

    label2 = tk.Label(new_f,
                      text='Please, provide the path of the folder with images to classify',
                      font=('Ubuntu Mono', 12))
    label2.config(bg='white')
    label2.pack(ipadx=10, ipady=10, expand=True)

    input_text2 = tk.Text(new_f, height=2, width=60)
    input_text2.pack(ipadx=10, ipady=10, expand=False)

    b = tk.Button(new_f, text='Start', height=1, width=30,
                  command=lambda: load_model(new_f, input_text2, input_text1))
    b.pack(ipadx=10, ipady=10, expand=True)


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
        f.pack_forget()
        new_f = tk.Frame(root, width=700, height=600)
        new_f.config(bg='white')
        new_f.pack(fill='both', expand=True)

        label1 = tk.Label(new_f,
                          text='Starting classification',
                          font=('Ubuntu Mono', 18))
        label1.config(bg='white')
        label1.pack(ipadx=10, ipady=30)

        model = Train(input_folder=None, output_folder=folder_to_classify, model_folder=model_folder)

        start_classification(new_f, model)


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
    label1.pack(ipadx=10, ipady=10, expand=True)

    input_text1 = tk.Text(new_f, height=2, width=60)
    input_text1.pack(ipadx=10, ipady=10, expand=False)

    label2 = tk.Label(new_f,
                      text='Please, provide the path of the folder with images to classify',
                      font=('Ubuntu Mono', 12))
    label2.config(bg='white')
    label2.pack(ipadx=10, ipady=10, expand=True)

    input_text2 = tk.Text(new_f, height=2, width=60)
    input_text2.pack(ipadx=10, ipady=10, expand=False)

    label3 = tk.Label(new_f,
                      text='Please, provide the path to save the model (existing or new)',
                      font=('Ubuntu Mono', 12))
    label3.config(bg='white')
    label3.pack(ipadx=10, ipady=10, expand=True)

    input_text3 = tk.Text(new_f, height=2, width=60)
    input_text3.pack(ipadx=10, ipady=10, expand=True)

    b = tk.Button(new_f, text='Start', height=1, width=30,
                  command=lambda: create_model(new_f, input_text1, input_text2, input_text3))
    b.pack(ipadx=10, ipady=10, expand=True)


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
    f.pack_forget()
    f = tk.Frame(root, width=700, height=600)
    f.config(bg='white')
    f.pack(fill='both', expand=True)

    label1 = tk.Label(f,
                      text='Starting training',
                      font=('Ubuntu Mono', 18))
    label1.config(bg='white')
    label1.pack(ipadx=10, ipady=30)

    trainer.train()

    #
    # add progress bar
    #
    # b = tk.Button(new_f, text='Start', command=f.quit())
    # b.pack()

    label2 = tk.Label(f, text='Training done!', font=("Ubuntu Mono", 18))
    label2.pack(ipadx=10, ipady=30)

    stop_or_continue(f, trainer)


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
    f.pack_forget()
    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    trainer.classify()
    label2 = tk.Label(new_f, text='Classification done! \n press the button to close', font=('Ubuntu Mono', 18))
    label2.config(bg='white')
    label2.pack(ipadx=10, ipady=30)

    b = tk.Button(new_f, text='Close', height=1, width=30,
                  command=lambda: root.quit())
    b.pack(ipadx=10, ipady=10, expand=True)

    #
    # add progress bar
    #
    # add exit button


def popup(f, text):
    new_f = tk.Frame(f, width=300, height=250)
    new_f.config(bg='gray')
    new_f.place(in_=f, anchor="c", relx=.5, rely=.5)

    l = tk.Label(new_f, text=text, font=('Ubuntu Mono', 12))
    l.config(bg='gray')
    l.pack(ipadx=10, ipady=30)

    close_b = tk.Button(new_f, text='Close', command=lambda: new_f.place_forget())
    close_b.config(bg='white')
    close_b.pack()


def popup_hippo(f):
    f.pack_forget()
    frame = tk.Frame(root, width=700, height=600)
    frame.config(bg='white')
    frame.pack(fill='both', expand=True)

    picture = tk.PhotoImage(file='hippo.png').subsample(5, 5)
    label = tk.Label(frame, text='Welcome!', font=('Ubuntu Mono', 18), image=picture, compound='bottom')
    label.image = picture
    label.config(bg='white')
    label.pack(ipadx=10, ipady=50)

    start_button = tk.Button(frame, text='Press here to start', command=lambda: choose_model(frame))
    start_button.pack(ipadx=5, ipady=5, expand=True)


start_frame(root)

root.mainloop()





