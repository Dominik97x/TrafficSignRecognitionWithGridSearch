import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
#wczytanie wytrenowanego modelu
from tensorflow.keras.models import  load_model
model = load_model('my_model.h5')

#słownik etykietujący każda klasę znaków.
classes = {
1:'Ograniczenie prędkości (20km/h)',
2:'Ograniczenie prędkości  (30km/h)',
3:'Ograniczenie prędkości  (50km/h)',
4:'Ograniczenie prędkości  (60km/h)',
5:'Ograniczenie prędkości (70km/h)',
6:'Ograniczenie prędkości  (80km/h)',
7:'Koniec ograniczenia prędkości (80km/h)',
8:'Ograniczenie prędkości (100km/h)',
9:'Ograniczenie prędkości (120km/h)',
10:'Zakaz wyprzedzania ',
11:'Zakaz wyprzedzania przez samochody ciężarowe',
12:'Skrzyżowanie z drogą podporządkowaną występującą po obu stronach',
13:'Droga z pierwszeństwem',
14:'Ustąp pierwszeństwa',
15:'Stop',
16:'Zakaz ruchu w obu kierunkach',
17:'Zakaz wjazdu samochodów ciężarowych',
18:'Zakaz wjazdu',
19:'Inne niebezpieczeństwo',
20:'Niebezpieczny zakręt w lewo',
21:'Niebezpieczny zakręt w prawo',
22:'Niebezpieczne zakręty, pierwszy w lewo',
23:'Nierówna droga',
24:'Śliska jezdnia',
25:'Zwężenie jezdni - prawostronne',
26:'Roboty drogowe',
27:'Sygnały świetlne',
28:'Przejście dla pieszych',
29:'Dzieci',
30:'Rowerzyści',
31:'Oszronienie jezdni',
32:'Zwierzęta dzikie',
33:'Koniec zakazów',
34:'Nakaz jazdy w prawo za znakiem',
35:'Nakaz jazdy w lewo za znakiem',
36:'Nakaz jazdy prosto',
37:'Nakaz jazdy prosto lub w prawo',
38:'Nakaz jazdy prosto lub w lewo',
39:'Nakaz jazdy z prawej strony znaku',
40:'Nakaz jazdy z lewej strony znaku',
41:'Ruch okrężny',
42:'Koniec zakazu wyprzedzania',
43:'Koniec zakazu wyprzedzania przez samochody ciężarowe' }
                 
#Inicjalizacja GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Rozpoznawania znaków drogowych')
top.configure(background='#eef3ef')
label=Label(top,background='#eef3ef', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#364156', text=sign,font=('sans',15,'bold'))
   

def show_classify_button(file_path):
    classify_b=Button(top,text="Rozpoznaj obraz",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Wgraj zdjęcie",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Rozpoznaj wybrany znak drogowy",pady=20, font=('arial',20,'bold'))
heading.configure(background='#eef3ef',foreground='#364156')
heading.pack()
top.mainloop()
