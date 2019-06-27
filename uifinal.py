"""
Alunos: 
Luis Felipe Gomes Costa - 20170002444
Pablo Ravelly Cotrim Teixeira
"""

import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from tkinter import filedialog
from tkinter import *

training_set = '/'
test_set = '/'

def load_directory():
    dirName = filedialog.askdirectory()
    return dirName

def load_file():
    fileName = filedialog.askopenfilename()
    return fileName

class Janela:



    def __init__(self,toplevel):
        self.frame=Frame(toplevel)
        self.frame.pack()
        self.frame2=Frame(toplevel)
        self.frame2.pack()
        self.titulo=Label(self.frame,text='Que tal tentarmos Classificar a Imagem do seu bichinho?',
                          font=('Verdana','18'))
        self.titulo.pack(pady = 30)
        self.msg=Label(self.frame,
                        text = 'Nós usamos uma CNN para reconhecimento de gatos e cachorros',
                        font=('Arial','12'))
        self.infor=Label(self.frame,
                        text = 'O resultado aparecerá aqui.',
                        font=('Arial','10'))
        self.msg.focus_force()
        self.msg.pack(pady = 20)
        self.infor.pack(pady = 150)
        # Definindo o botão 1
        self.b01=Button(self.frame2,text='Abrir pasta de treino...')
        self.b01['padx'],self.b01['pady'] = 10, 5
        self.b01['bg']='grey'
        self.b01.bind("<Return>",self.keypress01)
        self.b01.bind("<Any-Button>",self.button01)
        self.b01.bind("<FocusOut>",self.fout01)
        self.b01['relief']=RIDGE
        self.b01.pack(padx = 20, ipadx = 30, ipady = 30, side=LEFT)
        # Definindo o botão 2
        self.b02=Button(self.frame2,text='Abrir pasta de teste...')
        self.b02['padx'],self.b02['pady'] = 10, 5
        self.b02['bg']='grey'
        self.b02.bind("<Return>",self.keypress02)
        self.b02.bind("<Any-Button>",self.button02)
        self.b02.bind("<FocusIn>",self.fin02)
        self.b02.bind("<FocusOut>",self.fout02)
        self.b02['relief']=RIDGE
        self.b02.pack(padx = 20, ipadx = 30, ipady = 30, side=LEFT)
        # Definindo o botão 3
        self.b03=Button(self.frame2,text='Treinar!')
        self.b03['padx'],self.b03['pady'] = 10, 5
        self.b03['bg']='grey'
        self.b03.bind("<Return>",self.keypress03)
        self.b03.bind("<Any-Button>",self.button03)
        self.b03.bind("<FocusIn>",self.fin03)
        self.b03.bind("<FocusOut>",self.fout03)
        self.b03['relief']=RIDGE
        self.b03.pack(padx = 20, ipadx = 30, ipady = 30, side=LEFT)
        




    def keypress01(self,event): self.msg['text']='ENTER sobre o Botão 1'
    def keypress02(self,event): self.msg['text']='ENTER sobre o Botão 2'
    def keypress03(self,event): self.msg['text']='ENTER sobre o Botão 4'
    def button01(self,event):
        global training_set
        training_set = load_directory()
    def button02(self,event):
        global test_set
        test_set = load_directory()
    def fin01(self,event): self.b01['relief']=FLAT
    def fout01(self,event): self.b01['relief']=RIDGE
    def fin02(self,event): self.b02['relief']=FLAT
    def fout02(self,event): self.b02['relief']=RIDGEE
    def fin03(self,event): self.b03['relief']=FLAT
    def fout03(self,event): self.b03['relief']=RIDGE



    def button03(self,event):

        global training_set
        global test_set

        self.infor.config(text='estamos treinando... Por favor, aguarde!', foreground = 'red')

        if (training_set != '/' and test_set != '/'):

            classificador = Sequential()
            classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
            classificador.add(BatchNormalization())
            classificador.add(MaxPooling2D(pool_size = (2,2)))

            classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
            classificador.add(BatchNormalization())
            classificador.add(MaxPooling2D(pool_size = (2,2)))

            classificador.add(Flatten())

            classificador.add(Dense(units = 128, activation = 'relu'))
            classificador.add(Dropout(0.2))
            classificador.add(Dense(units = 128, activation = 'relu'))
            classificador.add(Dropout(0.2))
            classificador.add(Dense(units = 1, activation = 'sigmoid'))

            classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                                  metrics = ['accuracy'])

            gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                                     rotation_range = 7,
                                                     horizontal_flip = True,
                                                     shear_range = 0.2,
                                                     height_shift_range = 0.07,
                                                     zoom_range = 0.2)
            gerador_teste = ImageDataGenerator(rescale = 1./255)

            base_treinamento = gerador_treinamento.flow_from_directory(training_set,
                                                                       target_size = (64, 64),
                                                                       batch_size = 32,
                                                                       class_mode = 'binary')
            base_teste = gerador_teste.flow_from_directory(test_set,
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')

            classificador.fit_generator(base_treinamento, steps_per_epoch = 4000 / 32,
                                        epochs = 1, validation_data = base_teste,
                                        validation_steps = 1000 / 32)



            self.msg.config(text=classificador.metrics)

        else:
            self.infor.config(text='por favor selecione as pastas de treino e teste.', foreground = 'red')


raiz=Tk()
raiz.title("Classificador de imagens")


window_height = 600
window_width = 900

screen_width = raiz.winfo_screenwidth()
screen_height = raiz.winfo_screenheight()

x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))

raiz.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))


Janela(raiz)
raiz.mainloop() 