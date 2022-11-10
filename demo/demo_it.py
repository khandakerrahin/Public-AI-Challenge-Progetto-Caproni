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
    label = tk.Label(frame, text='Benvenuti!', font=('Ubuntu Mono', 18, 'bold'), image=picture, compound='bottom')
    label.image = picture
    label.config(bg='white')
    label.pack(ipadx=10, ipady=50)

    start_button = tk.Button(frame, text='Premi qui per iniziare',  height=2, width=15,
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


# select existing or new model
def choose_model(f):
    f.pack_forget()
    f = tk.Frame(root, width=700, height=600)
    f.config(bg='white')
    f.pack(fill='both', expand=True)

    label = tk.Label(f, text="Seleziona l'obiettivo", font=("Ubuntu Mono", 18, 'bold'))
    label.config(bg='white')
    label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    start_classification = tk.Button(f, text='Suddivisione Tematica', height=4, width=30,
                                     command=lambda: classification(f))
    start_classification.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    start_metadata = tk.Button(f, text='Estrazione di Metadata', height=4, width=30, command=lambda: metadata(f))
    start_metadata.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # start_segmentation = tk.Button(f, text='Damage Identification', height=5, width=30)
    # start_segmentation.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    menu_button = tk.Button(f, text='Indietro', height=2, width=10, command=lambda: start_frame(root, f))
    menu_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def classification(f):
    f.pack_forget()

    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    input_text = filedialog.askdirectory(initialdir="/", title="Seleziona la cartella delle immagini")

    label1 = tk.Label(new_f,
                      text=f'Cartella immagii selezionata: {input_text}',
                      font=('Ubuntu Mono', 15))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    b = tk.Button(new_f, text='Inizia', height=2, width=30,
                  command=lambda: start_classification(new_f, input_text))
    b.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    back_button = tk.Button(new_f, text='Indietro', height=2, width=10, command=lambda: choose_model(new_f))
    back_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def start_classification(f, folder_to_classify):
    model = Classify(input_folder=folder_to_classify)

    pb = threading.Thread(target=start_progress_bar,
                      args=(f, 'Procedo con la Suddivisione Tematica', model, 'c'))
    pb.start()


def classification_done(f):
    f.pack_forget()
    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    label1 = tk.Label(new_f, text="Suddivisione tematica terminata!", font=("Ubuntu Mono", 18, 'bold'))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label2 = tk.Label(new_f, text='premi Menu per andare alla pagina iniziale \n'
                                  'Premi Chiudi per terminare.                ',
                      font=('Ubuntu Mono', 16))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    b1 = tk.Button(new_f, text='Menu', height=2, width=30, command=lambda: start_frame(root, new_f))
    b1.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    b2 = tk.Button(new_f, text="Chiudi", height=2, width=30, command=lambda: root.quit())
    b2.place(relx=0.5, rely=0.8, anchor=tk.CENTER)


def metadata(f):
    f.pack_forget()

    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    input_text1 = filedialog.askdirectory(initialdir="/", title="Seleziona la cartella delle immagini")
    input_text2 = filedialog.askdirectory(initialdir="/", title="Seleziona la cartella dove salvare i risulati")

    label1 = tk.Label(new_f,
                      text=f'Cartella immagini selezionata: {input_text1}',
                      font=('Ubuntu Mono', 15))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label2 = tk.Label(new_f,
                      text=f'Cartella risultati selezionata: {input_text2}',
                      font=('Ubuntu Mono', 15))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    b = tk.Button(new_f, text='Inizia', height=2, width=30,
                  command=lambda: start_metadata_extraction(new_f, input_text1, input_text2))
    b.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    back_button = tk.Button(new_f, text='Indietro', height=2, width=10, command=lambda: choose_model(new_f))
    back_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def start_metadata_extraction(f, image_folder, output_folder, task='s'):
    ME = MetadataExtraction(image_folder, output_folder)
    pb = threading.Thread(target=start_progress_bar,
                          args=(f, "Procedo con l'estrazione dei metadata", ME, 'me'))
    pb.start()


def metadata_extraction_done(f):
    f.pack_forget()
    new_f = tk.Frame(root, width=700, height=600)
    new_f.config(bg='white')
    new_f.pack(fill='both', expand=True)

    label1 = tk.Label(new_f, text="Estrazione dei metadata terminata!", font=("Ubuntu Mono", 18, 'bold'))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label2 = tk.Label(new_f, text='premi Menu per andare alla pagina iniziale \n'
                                  'Premi Chiudi per terminare.                ',
                      font=('Ubuntu Mono', 16))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    b1 = tk.Button(new_f, text='Menu', height=2, width=30, command=lambda: start_frame(root, new_f))
    b1.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    b2 = tk.Button(new_f, text="Chiudi", height=2, width=30, command=lambda: root.quit())
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
    label = tk.Label(frame, text='Welcome!', font=('Ubuntu Mono', 18, 'bold'), image=picture, compound='bottom')
    label.image = picture
    label.config(bg='white')
    label.pack(ipadx=10, ipady=50)

    start_button = tk.Button(frame, text='Press here to start',  height=2, width=15,
                             command=lambda: choose_model(frame))
    start_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    info_button = tk.Button(frame, text='Info', height=2, width=15, command=lambda: get_info(frame))
    info_button.place(relx=0.5, rely=0.9, anchor=tk.CENTER)


def get_info(f):
    f.pack_forget()
    f = tk.Frame(root, width=700, height=600)
    f.config(bg='white')
    f.pack(fill='both', expand=True)

    info_title = tk.Label(f, text='Info',  font=('Ubuntu Mono', 18))
    info_title.config(bg='white')
    info_title.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    label1 = tk.Label(f, text='Suddivisione tematica                                                ',
                      font=('Ubuntu Mono', 14, 'bold'))
    label1.config(bg='white')
    label1.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    label2 = tk.Label(f, text="Seleziona la cartella con le immagini                                      \n"
                              "Alla fine del processo, una nuova cartella 'classification_result', verrà  \n"
                              "creata all'interno della cartella immagini. La nuova cartella conterrà     \n"
                              "le immagini suddivise in cartelle tematiche.                               \n\n"
                              "N.B le immagini saranno una copia delle originali.                         ",
                      anchor='w', font=("Ubuntu Mono", 12))
    label2.config(bg='white')
    label2.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    label3 = tk.Label(f, text="Estrazione dei Metadata                                         ",
                      font=("Ubuntu Mono", 14, " bold"))
    label3.config(bg='white')
    label3.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    label4 = tk.Label(f, text='Seleziona la cartella con le immagini                                      \n'
                              "Alla fine del processo, verrà creato un file CSV, metadata_results.csv,    \n"
                              "nella cartella indicata. Il file conterrà il path delle immagini, il loro  \n"
                              "soggeto e il contenuto.                                                    \n",
                      anchor='w', font=("Ubuntu Mono", 12))
    label4.config(bg='white')
    label4.place(relx=0.5, rely=0.6, anchor=tk.CENTER)

    # label5 = tk.Label(f, text="Developed by Ludovico Maria Valenti, Alessia Meloni, Shaker Mahmud Khandaker, "
    #                           "Md Ashraful Alam Hazari", font=('Ubuntu Mono', 9))
    # label5.config(bg='white')
    # label5.place(relx=0.5, rely=0.8, anchor=tk.CENTER)

    menu_button = tk.Button(f, text='Indietro', command=lambda: start_frame(root, f))
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

    l = tk.Label(f, text=text, font=("Ubuntu Mono", 18, 'bold'))
    l.config(bg='white')
    l.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

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

