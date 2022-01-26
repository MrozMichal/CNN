from tkinter import *
from tkinter import messagebox
from tkinter import filedialog as fd
from NeuralNetwork import NeuralNetwork as NN
import NewBP as NB
from matplotlib import pyplot as plt
import numpy as np
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import copy

global Icons_counter
Icons_counter=0

class TrainingWindow:
    def __init__(self, epochs, training_history):
        self.frame = Tk()
        self.frame.resizable(0, 0)
        self.frame.title('Przebieg uczenia sieci')
        screen_width = self.frame.winfo_screenwidth()
        screen_height = self.frame.winfo_screenheight()
        self.frame.geometry('325x235')
        #self.frame.geometry('325x235+%d+%d' % ((screen_width - 325) / 2, (screen_height - 235) / 2))
        self.frame.grid_propagate(False)
        self.frame.eval('tk::PlaceWindow . center')

        self.epochs = epochs
        self.training_history = training_history

        self.__create_widgets()

        self.frame.mainloop()


    def __create_widgets(self):
        self.training_label = Label(self.frame, text='Metoda uczenie:     Spadek wzdłuż gradientu z momentem')
        self.training_label.place(x=5, y=5)

        self.cost_label = Label(self.frame, text='Funkcja kosztu:       Błąd średniokwadratowy')
        self.cost_label.place(x=5, y=25)

        self.epochs_label = Label(self.frame, text='Liczba epok:            {:d}'.format(self.epochs))
        self.epochs_label.place(x=5, y=45)

        self.plot_performance_button = Button(self.frame, text='Wykres trafności sieci', bg='#E1E1E1',
                                              command=lambda: self.__plot_performance())
        self.plot_performance_button.place(x=5, y=120, width=315, height=50)

        self.plot_loss_button = Button(self.frame, text='Wykres kosztu sieci', bg='#E1E1E1',
                                       command=lambda: self.__plot_loss())
        self.plot_loss_button.place(x=5, y=180, width=315, height=50)

    def __plot_performance(self):
        history_dict = self.training_history.history

        plt.figure()
        plt.plot(history_dict['accuracy'])
        plt.xlabel('{:d} Epok'.format(self.epochs))
        plt.ylabel('trafności')
        plt.title('Wykres trafności sieci')
        plt.grid(True)
        plt.show()

    def __plot_loss(self):
        history_dict = self.training_history.history

        plt.figure()
        plt.plot(history_dict['loss'])
        plt.xlabel('{:d} Epok'.format(self.epochs))
        plt.ylabel('Błąd średniokwadratowy')
        plt.title('Wykres błędu sieci')
        plt.grid(True)
        plt.show()


class Gui:
    def __init__(self):
        self.frame = Tk()
        self.frame.resizable(0, 0)
        self.frame.title('Back Propagation GUI')
        screen_width = self.frame.winfo_screenwidth()
        screen_height = self.frame.winfo_screenheight()
        self.frame.geometry('1170x660')
        #self.frame.geometry('1170x660+%d+%d' % ((screen_width - 1170) / 2, (screen_height - 660) / 2))
        self.frame.grid_propagate(False)
        self.train_dataset = None
        self.train_dataset2 = None
        self.train_dataset3 = None
        self.test_dataset = None
        self.weights = []

        # Ustawianie stanu wczytania pliku uczacego i testowego
        self.train_var_state = False
        self.test_var_state = False

        self.__create_widgets()
        self.network = None
        self.training_window = None
        self.frame.mainloop()

    def Ikona(self):
        global Icons_counter
        Icons_counter += 1
        # self.number_of_random_wages_label = Label(self.standard_parameters_frame, text='Zakres losowania wag:')
        # self.number_of_random_wages_label.place(x=5, y=15, height=25)
        # self.number_of_random_wages_var = StringVar()
        # self.acctual_icon='1'
        # self.number_of_random_wages_entry = Entry(self.standard_parameters_frame, justify=CENTER,
        #                                           textvariable=self.acctual_icon)
        # self.number_of_random_wages_entry.insert(0, '0.1')
        # self.number_of_random_wages_entry.place(x=145, y=15, width=50, height=25)
        if self.train_dataset is None: # TO jest inicjalizowane w NN
            return messagebox.showerror('Nie podano pliku', 'Wczytaj plik uczacy')
        Icons = ['1', '2', '3']
        if (Icons_counter == 3):
            Icons_counter = 0
        self.Ikona_output_var = StringVar(value=Icons[Icons_counter])
        self.Ikona_output_entry = Entry(self.frame, justify=CENTER, textvariable=self.Ikona_output_var,
                                        readonlybackground='#FFFFFF')
        self.Ikona_output_entry.place(x=300, y=600, width=75, height=30)

        # ICONS
        self.ikona1 = int(self.train_dataset[Icons_counter][0])
        self.ikona2 = int(self.train_dataset[Icons_counter][1])
        self.ikona3 = int(self.train_dataset[Icons_counter][2])
        self.ikona4 = int(self.train_dataset[Icons_counter][3])
        self.ikona5 = int(self.train_dataset[Icons_counter][4])
        self.ikona6 = int(self.train_dataset[Icons_counter][5])
        self.ikona7 = int(self.train_dataset[Icons_counter][6])
        self.ikona8 = int(self.train_dataset[Icons_counter][7])
        self.ikona9 = int(self.train_dataset[Icons_counter][8])

        self.ikonaAnswer1 = StringVar()
        self.ikonaAnswer2 = StringVar()
        self.ikonaAnswer3 = StringVar()

        self.ikona1 = Entry(self.frame, justify=CENTER,
                            textvariable=StringVar(value=self.ikona1),
                            readonlybackground='#FFFFFF')
        self.ikona2 = Entry(self.frame, justify=CENTER,
                            textvariable=StringVar(value=self.ikona2),
                            readonlybackground='#FFFFFF')
        self.ikona3 = Entry(self.frame, justify=CENTER,
                            textvariable=StringVar(value=self.ikona3),
                            readonlybackground='#FFFFFF')
        self.ikona4 = Entry(self.frame, justify=CENTER,
                            textvariable=StringVar(value=self.ikona4),
                            readonlybackground='#FFFFFF')
        self.ikona5 = Entry(self.frame, justify=CENTER,
                            textvariable=StringVar(value=self.ikona5),
                            readonlybackground='#FFFFFF')
        self.ikona6 = Entry(self.frame, justify=CENTER,
                            textvariable=StringVar(value=self.ikona6),
                            readonlybackground='#FFFFFF')
        self.ikona7 = Entry(self.frame, justify=CENTER,
                            textvariable=StringVar(value=self.ikona7),
                            readonlybackground='#FFFFFF')
        self.ikona8 = Entry(self.frame, justify=CENTER,
                            textvariable=StringVar(value=self.ikona8),
                            readonlybackground='#FFFFFF')
        self.ikona9 = Entry(self.frame, justify=CENTER,
                            textvariable=StringVar(value=self.ikona9),
                            readonlybackground='#FFFFFF')

        self.ikonaAnswer1 = Entry(self.frame, justify=CENTER,
                                  textvariable=self.ikonaAnswer1,
                                  state='readonly', readonlybackground='#FFFFFF')
        self.ikonaAnswer2 = Entry(self.frame, justify=CENTER,
                                  textvariable=self.ikonaAnswer2,
                                  state='readonly', readonlybackground='#FFFFFF')
        self.ikonaAnswer3 = Entry(self.frame, justify=CENTER,
                                  textvariable=self.ikonaAnswer3,
                                  state='readonly', readonlybackground='#FFFFFF')

        self.ikona1.place(x=120, y=425, width=40, height=40)
        self.ikona2.place(x=190, y=425, width=40, height=40)
        self.ikona3.place(x=260, y=425, width=40, height=40)
        self.ikona4.place(x=120, y=475, width=40, height=40)
        self.ikona5.place(x=190, y=475, width=40, height=40)
        self.ikona6.place(x=260, y=475, width=40, height=40)
        self.ikona7.place(x=120, y=525, width=40, height=40)
        self.ikona8.place(x=190, y=525, width=40, height=40)
        self.ikona9.place(x=260, y=525, width=40, height=40)

        self.ikonaAnswer1.place(x=470, y=425, width=40, height=40)
        self.ikonaAnswer2.place(x=470, y=475, width=40, height=40)
        self.ikonaAnswer3.place(x=470, y=525, width=40, height=40)

    def __create_widgets(self):
        self.__create_data_panel_widgets()
        self.__create_standard_parameters_widgets()
        self.__create_optional_parameters_widgets()
        self.__create_action_panel_widgets()
        self.__create_visualization_widgets()
        self.__create_visualization_options_widgets()

        self.response_surface_button = Button(self.frame, text='Ikona', bg='#E1E1E1',
                                              command=lambda: self.Ikona())
        self.response_surface_button.place(x=180, y=600, width=100, height=30)

        #self.Ikona_output_label = Label(self.frame, text='Wejścia')
        #self.Ikona_output_label.place(x=320, y=600, width=100, height=25)

        self.Ikona_output_var = StringVar()
        self.Ikona_output_entry = Entry(self.frame, justify=CENTER,
                                               textvariable=self.Ikona_output_var,
                                               state='readonly', readonlybackground='#FFFFFF')
        self.Ikona_output_entry.place(x=300, y=600, width=75, height=30)
        self.Ikona_output_entry.insert(0, '0')


        # ICONS
        self.ikona1 = StringVar()
        self.ikona2 = StringVar()
        self.ikona3 = StringVar()
        self.ikona4 = StringVar()
        self.ikona5 = StringVar()
        self.ikona6 = StringVar()
        self.ikona7 = StringVar()
        self.ikona8 = StringVar()
        self.ikona9 = StringVar()

        self.ikonaAnswer1 = StringVar()
        self.ikonaAnswer2 = StringVar()
        self.ikonaAnswer3 = StringVar()

        self.ikona1 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikona1,
                                       state='readonly', readonlybackground='#FFFFFF')
        self.ikona2 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikona2,
                                       state='readonly', readonlybackground='#FFFFFF')
        self.ikona3 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikona3,
                                       state='readonly', readonlybackground='#FFFFFF')
        self.ikona4 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikona4,
                                       state='readonly', readonlybackground='#FFFFFF')
        self.ikona5 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikona5,
                                       state='readonly', readonlybackground='#FFFFFF')
        self.ikona6 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikona6,
                                       state='readonly', readonlybackground='#FFFFFF')
        self.ikona7 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikona7,
                                       state='readonly', readonlybackground='#FFFFFF')
        self.ikona8 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikona8,
                                       state='readonly', readonlybackground='#FFFFFF')
        self.ikona9 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikona9,
                                       state='readonly', readonlybackground='#FFFFFF')

        self.ikonaAnswer1 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikonaAnswer1,
                                       state='readonly', readonlybackground='#FFFFFF')
        self.ikonaAnswer2 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikonaAnswer2,
                                       state='readonly', readonlybackground='#FFFFFF')
        self.ikonaAnswer3 = Entry(self.frame, justify=CENTER,
                                       textvariable=self.ikonaAnswer3,
                                       state='readonly', readonlybackground='#FFFFFF')

        self.ikona1.place(x=120, y=425, width=40, height=40)
        self.ikona2.place(x=190, y=425, width=40, height=40)
        self.ikona3.place(x=260, y=425, width=40, height=40)
        self.ikona4.place(x=120, y=475, width=40, height=40)
        self.ikona5.place(x=190, y=475, width=40, height=40)
        self.ikona6.place(x=260, y=475, width=40, height=40)
        self.ikona7.place(x=120, y=525, width=40, height=40)
        self.ikona8.place(x=190, y=525, width=40, height=40)
        self.ikona9.place(x=260, y=525, width=40, height=40)

        self.ikonaAnswer1.place(x=470, y=425, width=40, height=40)
        self.ikonaAnswer2.place(x=470, y=475, width=40, height=40)
        self.ikonaAnswer3.place(x=470, y=525, width=40, height=40)

        #self.__create_network_widgets()

    def Dataset_message(self):
        return messagebox.showinfo('Message', f'Dataset from path: {self.filename} has been loaded.')

    def read_train_csv(self):

        # Convert string column to float
        def str_column_to_float(dataset, column):
            for row in dataset:
                row[column] = float(row[column].strip())

        # Convert string column to integer
        def str_column_to_int(dataset, column):
            # class_values = [row[column] for row in dataset]
            # print('Class_values: ')
            # print(class_values)
            # unique = set(class_values)
            #
            # sorted(unique)
            # print(unique)
            # lookup = dict()
            # for i, value in enumerate(unique):
            #     lookup[value] = i
            for row in dataset:
                # row[column] = lookup[row[column]]
                row[column] = int(row[column])
            # print("Lookup:")
            # print(lookup)
            # return lookup

        self.filename = fd.askopenfilename()
        #print(self.filename)
        self.train_dataset = list()
        with open(self.filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                self.train_dataset.append(row)
            print("READING Train Dataset accomplished.")
            self.data_panel_train_entry.insert(0,f'{self.filename}')
            self.train_var_state = True
            self.Dataset_message()


        # MAIN LOOP in GUI
        for i in range(len(self.train_dataset[0]) - 1):
            str_column_to_float(self.train_dataset, i)

        # convert class column to integers
        str_column_to_int(self.train_dataset, len(self.train_dataset[0]) - 1)

        # making dataset 2
        self.train_dataset2 = copy.deepcopy(self.train_dataset)
        for i in range(len(self.train_dataset2)):
            if i == 1:
                self.train_dataset2[i][-1] = 1
            else:
                self.train_dataset2[i][-1] = 0

        # making dataset 3
        self.train_dataset3 = copy.deepcopy(self.train_dataset)
        for i in range(len(self.train_dataset3)):
            if i == 2:
                self.train_dataset3[i][-1] = 1
            else:
                self.train_dataset3[i][-1] = 0

        self.check_box_data_panel_train_entry.select()

    def read_test_csv(self):
        self.filename = fd.askopenfilename()
        #print(self.filename)
        self.test_dataset = list()
        with open(self.filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                self.test_dataset.append(row)
            print("READING TEST Dataset accomplished.")
            self.data_panel_test_entry.insert(0,f'{self.filename}')
            self.test_var_state = True
            self.Dataset_message()

        self.check_box_data_panel_test_entry.select()

    def __create_data_panel_widgets(self):
        self.data_panel_frame = LabelFrame(self.frame, text='Panel danych', width=500, height=100)
        self.data_panel_frame.place(x=45, y=15)

        # Wczytywanie pliku uczącego
        self.data_panel_train_button = Button(self.data_panel_frame, text='Wczytaj plik uczący', bg='#E1E1E1',
                                              command=lambda: self.read_train_csv())
        self.data_panel_train_button.place(x=15, y=10, width=120, height=25)
        self.data_panel_train_var = StringVar()
        self.data_panel_train_entry = Entry(self.data_panel_frame, justify=LEFT,
                                               textvariable=self.data_panel_train_var, readonlybackground='#FFFFFF')
        self.data_panel_train_entry.place(x=155, y=10, width=290, height=25)
        self.check_box_data_panel_train_entry = Checkbutton(self.data_panel_frame, state=DISABLED)
        self.check_box_data_panel_train_entry.deselect()
        self.check_box_data_panel_train_entry.place(x=445, y=10)


        # Wczytywanie pliku testowego
        self.data_panel_test_button = Button(self.data_panel_frame, text='Wczytaj plik testowy', bg='#E1E1E1',
                                              command=lambda: self.read_test_csv())
        self.data_panel_test_button.place(x=15, y=45, width=120, height=25)
        self.data_panel_test_var = StringVar()
        self.data_panel_test_entry = Entry(self.data_panel_frame, justify=LEFT,
                                               textvariable=self.data_panel_test_var,readonlybackground='#FFFFFF')
        self.data_panel_test_entry.place(x=155, y=45, width=290, height=25)
        self.check_box_data_panel_test_entry = Checkbutton(self.data_panel_frame, state=DISABLED)
        self.check_box_data_panel_test_entry.deselect()
        self.check_box_data_panel_test_entry.place(x=445, y=45)

    def __create_standard_parameters_widgets(self):
        self.standard_parameters_frame = LabelFrame(self.frame, text='Parametry standardowe', width=230, height=150)
        self.standard_parameters_frame.place(x=45, y=125)

        self.number_of_random_wages_label = Label(self.standard_parameters_frame, text='Zakres losowania wag:')
        self.number_of_random_wages_label.place(x=5, y=15, height=25)
        self.number_of_random_wages_var = StringVar()
        self.number_of_random_wages_entry = Entry(self.standard_parameters_frame, justify=CENTER,textvariable=self.number_of_random_wages_var)
        self.number_of_random_wages_entry.insert(0, '0.1')
        self.number_of_random_wages_entry.place(x=145, y=15, width=50, height=25)

        self.number_of_epochs_label = Label(self.standard_parameters_frame, text='Ilość epok:')
        self.number_of_epochs_label.place(x=5, y=45, height=25)
        self.number_of_epochs_var = StringVar()
        self.number_of_epochs_entry = Entry(self.standard_parameters_frame, justify=CENTER,textvariable=self.number_of_epochs_var)
        self.number_of_epochs_entry.insert(0, '500')
        self.number_of_epochs_entry.place(x=145, y=45, width=50, height=25)

        self.learning_rate_label = Label(self.standard_parameters_frame, text='Współczynnik uczenia:')
        self.learning_rate_label.place(x=5, y=75, height=25)
        self.learning_rate_var = StringVar()
        self.learning_rate_entry = Entry(self.standard_parameters_frame, justify=CENTER,textvariable=self.learning_rate_var)
        self.learning_rate_entry.insert(0, '0.9')
        self.learning_rate_entry.place(x=145, y=75, width=50, height=25)

    def __create_optional_parameters_widgets(self):
        self.optional_parameters_frame = LabelFrame(self.frame, text='Parametry opcjonalne', width=230, height=150)
        self.optional_parameters_frame.place(x=315, y=125)

        self.optional_parameter_ind = 0
        self.optional_parameter = ['BIAS', 'Współcz. Momentum']
        self.optional_parameter_var = IntVar()

        self.radio_button_bias = Radiobutton(self.optional_parameters_frame, text='BIAS', variable=self.optional_parameter_var, value=0)
        self.radio_button_bias.place(x=5, y=15)

        self.radio_button_momentum = Radiobutton(self.optional_parameters_frame, text='Współcz. Momentum', variable=self.optional_parameter_var, value=1)
        self.radio_button_momentum.place(x=5, y=45)

        self.momentum_var = IntVar()
        self.momentum_entry = Entry(self.optional_parameters_frame, justify=CENTER,textvariable=self.momentum_var)
        self.momentum_entry.insert(0, '0.6')
        self.momentum_entry.place(x=155, y=45, width=50, height=25)

    def __create_action_panel_widgets(self):
        self.optional_parameters_frame = LabelFrame(self.frame, text='Panel Akcji', width=500, height=100)
        self.optional_parameters_frame.place(x=45, y=285)

        self.action_panel_initialize_button = Button(self.optional_parameters_frame, text='Inicjalizuj sieć', bg='#E1E1E1', command=lambda: self.__initial_network())
        self.action_panel_initialize_button.place(x=15, y=15, width=90, height=40)
        self.action_panel_initialize_var = StringVar()
        self.check_box_initialize = Checkbutton(self.optional_parameters_frame, variable=self.action_panel_initialize_var, state=DISABLED)
        self.check_box_initialize.deselect()
        self.check_box_initialize.place(x=115, y=25)

        self.action_panel_learn_button = Button(self.optional_parameters_frame, text='Naucz sieć', bg='#E1E1E1', command=lambda: self.__train_network())
        self.action_panel_learn_button.place(x=200, y=15, width=90, height=40)

        self.action_panel_reset_button = Button(self.optional_parameters_frame, text='RESET', bg='#E1E1E1', command=lambda: self.__plot_response_surface())
        self.action_panel_reset_button.place(x=390, y=15, width=90, height=40)

    def __create_visualization_widgets(self):
        self.visualization_frame = LabelFrame(self.frame, text='Wizualizacja',width=500, height=550)
        #self.visualization_frame = LabelFrame(self.frame, text='Wizualizacja',width=500, height=850)
        self.visualization_frame.place(x=620, y=15,width=500, height=550)

        self.c = Canvas(self.visualization_frame)
        self.c.pack(fill=BOTH, expand=1)
        #outline="#ffff00", fill="#ffff00")   # żółty
        #outline="#ffffff", fill="#ffffff")   # biały
        #outline="#808080", fill="#808080")   # szary
        #outline="#000000", fill="#000000")   # czarny

        coords = [[0, 0],[52, 0],[104, 0],[0, 52],[52, 52],[104, 52],[0, 104],[52, 104],[104, 104]]
        offset = [0, 180, 360]
        width = 50

        #kwadraty
        for of in offset:
            for co in coords:
                self.c.create_rectangle(co[0], co[1]+of, co[0]+width, co[1]+width+of, outline="#000000", fill="#ffffff")

        ## WYNIKI LICZBOWE

        # IV kwadrat
        self.visualization_q11 = StringVar()
        self.visualization_q12 = StringVar()
        self.visualization_q13 = StringVar()
        self.visualization_q14 = StringVar()
        self.visualization_q15 = StringVar()
        self.visualization_q16 = StringVar()
        self.visualization_q17 = StringVar()
        self.visualization_q18 = StringVar()
        self.visualization_q19 = StringVar()

        self.visualization_q11 = Entry(self.visualization_frame, justify=CENTER,
                                            textvariable=self.visualization_q11, readonlybackground='#FFFFFF')
        self.visualization_q12 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q12, readonlybackground='#FFFFFF')
        self.visualization_q13 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q13, readonlybackground='#FFFFFF')
        self.visualization_q14 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q14, readonlybackground='#FFFFFF')
        self.visualization_q15 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q15, readonlybackground='#FFFFFF')
        self.visualization_q16 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q16, readonlybackground='#FFFFFF')
        self.visualization_q17 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q17, readonlybackground='#FFFFFF')
        self.visualization_q18 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q18, readonlybackground='#FFFFFF')
        self.visualization_q19 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q19, readonlybackground='#FFFFFF')

        self.visualization_q11.place(x=0 + 200, y=0, width=50, height=50)
        self.visualization_q12.place(x=0 + 252, y=0, width=50, height=50)
        self.visualization_q13.place(x=0 + 304, y=0, width=50, height=50)
        self.visualization_q14.place(x=0 + 200, y=52, width=50, height=50)
        self.visualization_q15.place(x=0 + 252, y=52, width=50, height=50)
        self.visualization_q16.place(x=0 + 304, y=52, width=50, height=50)
        self.visualization_q17.place(x=0 + 200, y=104, width=50, height=50)
        self.visualization_q18.place(x=0 + 252, y=104, width=50, height=50)
        self.visualization_q19.place(x=0 + 304, y=104, width=50, height=50)

        # V kwadrat
        self.visualization_q21 = StringVar()
        self.visualization_q22 = StringVar()
        self.visualization_q23 = StringVar()
        self.visualization_q24 = StringVar()
        self.visualization_q25 = StringVar()
        self.visualization_q26 = StringVar()
        self.visualization_q27 = StringVar()
        self.visualization_q28 = StringVar()
        self.visualization_q29 = StringVar()

        self.visualization_q21 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q21, readonlybackground='#FFFFFF')
        self.visualization_q22 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q22, readonlybackground='#FFFFFF')
        self.visualization_q23 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q23, readonlybackground='#FFFFFF')
        self.visualization_q24 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q24, readonlybackground='#FFFFFF')
        self.visualization_q25 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q25, readonlybackground='#FFFFFF')
        self.visualization_q26 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q26, readonlybackground='#FFFFFF')
        self.visualization_q27 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q27, readonlybackground='#FFFFFF')
        self.visualization_q28 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q28, readonlybackground='#FFFFFF')
        self.visualization_q29 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q29, readonlybackground='#FFFFFF')

        self.visualization_q21.place(x=0 + 200, y=0+180, width=50, height=50)
        self.visualization_q22.place(x=0 + 252, y=0+180, width=50, height=50)
        self.visualization_q23.place(x=0 + 304, y=0+180, width=50, height=50)
        self.visualization_q24.place(x=0 + 200, y=52+180, width=50, height=50)
        self.visualization_q25.place(x=0 + 252, y=52+180, width=50, height=50)
        self.visualization_q26.place(x=0 + 304, y=52+180, width=50, height=50)
        self.visualization_q27.place(x=0 + 200, y=104+180, width=50, height=50)
        self.visualization_q28.place(x=0 + 252, y=104+180, width=50, height=50)
        self.visualization_q29.place(x=0 + 304, y=104+180, width=50, height=50)

        # VI kwadrat
        self.visualization_q31 = StringVar()
        self.visualization_q32 = StringVar()
        self.visualization_q33 = StringVar()
        self.visualization_q34 = StringVar()
        self.visualization_q35 = StringVar()
        self.visualization_q36 = StringVar()
        self.visualization_q37 = StringVar()
        self.visualization_q38 = StringVar()
        self.visualization_q39 = StringVar()

        self.visualization_q31 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q31, readonlybackground='#FFFFFF')
        self.visualization_q32 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q32, readonlybackground='#FFFFFF')
        self.visualization_q33 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q33, readonlybackground='#FFFFFF')
        self.visualization_q34 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q34, readonlybackground='#FFFFFF')
        self.visualization_q35 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q35, readonlybackground='#FFFFFF')
        self.visualization_q36 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q36, readonlybackground='#FFFFFF')
        self.visualization_q37 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q37, readonlybackground='#FFFFFF')
        self.visualization_q38 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q38, readonlybackground='#FFFFFF')
        self.visualization_q39 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q39, readonlybackground='#FFFFFF')

        self.visualization_q31.place(x=0 + 200, y=0+360, width=50, height=50)
        self.visualization_q32.place(x=0 + 252, y=0+360, width=50, height=50)
        self.visualization_q33.place(x=0 + 304, y=0+360, width=50, height=50)
        self.visualization_q34.place(x=0 + 200, y=52+360, width=50, height=50)
        self.visualization_q35.place(x=0 + 252, y=52+360, width=50, height=50)
        self.visualization_q36.place(x=0 + 304, y=52+360, width=50, height=50)
        self.visualization_q37.place(x=0 + 200, y=104+360, width=50, height=50)
        self.visualization_q38.place(x=0 + 252, y=104+360, width=50, height=50)
        self.visualization_q39.place(x=0 + 304, y=104+360, width=50, height=50)


        # Neuron Answers
        self.visualization_N1 = StringVar()
        self.visualization_N2 = StringVar()
        self.visualization_N3 = StringVar()

        self.visualization_N1Active = StringVar()
        self.visualization_N2Active = StringVar()
        self.visualization_N3Active = StringVar()

        self.visualization_N1 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_N1,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_N2 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_N2,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_N3 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_N3,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_N1Active = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_N1Active,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_N2Active = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_N2Active,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_N3Active = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_N3Active,
                                           state='readonly', readonlybackground='#FFFFFF')

        self.visualization_N1_label = Label(self.visualization_frame, text='Wyjście 1 neuronu')
        self.visualization_N1_label.place(x=20 + 356, y=0, height=25)
        self.visualization_N1.place(x=50 + 356, y=26, width=50, height=50)
        self.visualization_N1Active.place(x=50 + 356, y=78, width=50, height=50)
        self.visualization_N1Active_label = Label(self.visualization_frame, text='Aktywacja 1 neuronu')
        self.visualization_N1Active_label.place(x=20 + 356, y=78+50, height=25)

        self.visualization_N2_label = Label(self.visualization_frame, text='Wyjście 2 neuronu')
        self.visualization_N2_label.place(x=20 + 356, y=180, height=25)
        self.visualization_N2.place(x=50 + 356, y=26+180, width=50, height=50)
        self.visualization_N2Active.place(x=50 + 356, y=78+180, width=50, height=50)
        self.visualization_N2Active_label = Label(self.visualization_frame, text='Aktywacja 2 neuronu')
        self.visualization_N2Active_label.place(x=20 + 356, y=78+180+50, height=25)

        self.visualization_N3_label = Label(self.visualization_frame, text='Wyjście 3 neuronu')
        self.visualization_N3_label.place(x=20 + 356, y=360, height=25)
        self.visualization_N3.place(x=50 + 356, y=26+360, width=50, height=50)
        self.visualization_N3Active.place(x=50 + 356, y=78+360, width=50, height=50)
        self.visualization_N3Active_label = Label(self.visualization_frame, text='Aktywacja 3 neuronu')
        self.visualization_N3Active_label.place(x=20 + 356, y=78+360+50, height=25)

        self.c.pack(fill=BOTH, expand=1)

    def __create_visualization_options_widgets(self):
        self.visualization_options_frame = LabelFrame(self.frame, text='Opcje wyświetlania', width=500, height=70)
        self.visualization_options_frame.place(x=620, y=580)

        self.number_of_actualization_label = Label(self.visualization_options_frame, text='Częstość aktualizacji wag:')
        self.number_of_actualization_label.place(x=55, y=15, height=25)
        self.number_of_actualization_var = StringVar()
        self.number_of_actualization_entry = Entry(self.visualization_options_frame, justify=CENTER,textvariable=self.number_of_actualization_var)
        self.number_of_actualization_entry.insert(0, '10')
        self.number_of_actualization_entry.place(x=250, y=15, width=70, height=25)

    def __initial_network(self):
        self.random_wages = float(self.number_of_random_wages_var.get())
        self.num_epoch = float(self.number_of_epochs_var.get())
        self.lear_rate = float(self.learning_rate_var.get())
        #self.n_epoch=int(self.num_epoch)
        self.n_epoch = 10000

        if self.train_dataset == None or self.train_dataset2 == None or self.train_dataset3 == None:
            return messagebox.showerror('Message', f'Network CANNOT be Initialized. Read the train file first.')
        else:
            self.network1, self.output_wages1 = NN.initialize_network(self)
            # #print(self.network1)
            # print(self.output_wages1)
            self.network2, self.output_wages2 = NN.initialize_network(self)
            self.network3, self.output_wages3 = NN.initialize_network(self)

            self.visualization_q11.insert(0,f'{round(self.output_wages1[0],3)}')
            self.visualization_q12.insert(0,f'{round(self.output_wages1[1],3)}')
            self.visualization_q13.insert(0,f'{round(self.output_wages1[2],3)}')
            self.visualization_q14.insert(0,f'{round(self.output_wages1[3],3)}')
            self.visualization_q15.insert(0,f'{round(self.output_wages1[4],3)}')
            self.visualization_q16.insert(0,f'{round(self.output_wages1[5],3)}')
            self.visualization_q17.insert(0,f'{round(self.output_wages1[6],3)}')
            self.visualization_q18.insert(0,f'{round(self.output_wages1[7],3)}')
            self.visualization_q19.insert(0,f'{round(self.output_wages1[8],3)}')

            self.visualization_q21.insert(0,f'{round(self.output_wages2[0],3)}')
            self.visualization_q22.insert(0,f'{round(self.output_wages2[1],3)}')
            self.visualization_q23.insert(0,f'{round(self.output_wages2[2],3)}')
            self.visualization_q24.insert(0,f'{round(self.output_wages2[3],3)}')
            self.visualization_q25.insert(0,f'{round(self.output_wages2[4],3)}')
            self.visualization_q26.insert(0,f'{round(self.output_wages2[5],3)}')
            self.visualization_q27.insert(0,f'{round(self.output_wages2[6],3)}')
            self.visualization_q28.insert(0,f'{round(self.output_wages2[7],3)}')
            self.visualization_q29.insert(0,f'{round(self.output_wages2[8],3)}')

            self.visualization_q31.insert(0,f'{round(self.output_wages3[0],3)}')
            self.visualization_q32.insert(0,f'{round(self.output_wages3[1],3)}')
            self.visualization_q33.insert(0,f'{round(self.output_wages3[2],3)}')
            self.visualization_q34.insert(0,f'{round(self.output_wages3[3],3)}')
            self.visualization_q35.insert(0,f'{round(self.output_wages3[4],3)}')
            self.visualization_q36.insert(0,f'{round(self.output_wages3[5],3)}')
            self.visualization_q37.insert(0,f'{round(self.output_wages3[6],3)}')
            self.visualization_q38.insert(0,f'{round(self.output_wages3[7],3)}')
            self.visualization_q39.insert(0,f'{round(self.output_wages3[8],3)}')

            self.check_box_initialize.select()
            return messagebox.showinfo('Message', f'Network has been INITIALIZED.')

    def __train_network(self):
        self.random_wages = float(self.number_of_random_wages_var.get())
        self.num_epoch = int(self.number_of_epochs_var.get())
        self.lear_rate = float(self.learning_rate_var.get())

        if self.network1 is None: # TO jest inicjalizowane w NN
            return messagebox.showerror('Sieć nie istnieje', 'Przed rozpoczęciem trenowania sieci należy ją utworzyć')
        else:
            self.n_folds = 1
            self.bias = 0
            self.momentum = 0
            #self.folds = NN.cross_validation_split(self)

            n_folds = 1
            l_rate = 0.9
            n_epoch = 10000
            n_hidden = 1
            momentum = 0
            bias = 0

            scores = NB.evaluate_algorithm(self,self.train_dataset, NB.back_propagation, self.n_folds,self.lear_rate,self.num_epoch, n_hidden,rand_range=self.random_wages,momentum=self.momentum,bias=self.bias)

            self.visualization_q11.delete(0, "end")
            self.visualization_q11.insert(0, f'{round(self.output_wages1[0], 3)}')
            self.visualization_q12.delete(0, "end")
            self.visualization_q12.insert(0, f'{round(self.output_wages1[1], 3)}')
            self.visualization_q13.delete(0, "end")
            self.visualization_q13.insert(0, f'{round(self.output_wages1[2], 3)}')
            self.visualization_q14.delete(0, "end")
            self.visualization_q14.insert(0, f'{round(self.output_wages1[3], 3)}')
            self.visualization_q15.delete(0, "end")
            self.visualization_q15.insert(0, f'{round(self.output_wages1[4], 3)}')
            self.visualization_q16.delete(0, "end")
            self.visualization_q16.insert(0, f'{round(self.output_wages1[5], 3)}')
            self.visualization_q17.delete(0, "end")
            self.visualization_q17.insert(0, f'{round(self.output_wages1[6], 3)}')
            self.visualization_q18.delete(0, "end")
            self.visualization_q18.insert(0, f'{round(self.output_wages1[7], 3)}')
            self.visualization_q19.delete(0, "end")
            self.visualization_q19.insert(0, f'{round(self.output_wages1[8], 3)}')

            for i in range(0, len(self.output_wages1)-1):
                self.weights.append(self.output_wages1[i])

            print(self.output_wages1)

            scores2 = NB.evaluate_algorithm(self, self.train_dataset2, NB.back_propagation, n_folds, l_rate, n_epoch,n_hidden, rand_range=0.1, momentum=0, bias=0)

            # print('Scores: %s' % scores)
            # print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
            # print(self.output_wages1)
            self.visualization_q21.delete(0, "end")
            self.visualization_q21.insert(0, f'{round(self.output_wages1[0], 3)}')
            self.visualization_q22.delete(0, "end")
            self.visualization_q22.insert(0, f'{round(self.output_wages1[1], 3)}')
            self.visualization_q23.delete(0, "end")
            self.visualization_q23.insert(0, f'{round(self.output_wages1[2], 3)}')
            self.visualization_q24.delete(0, "end")
            self.visualization_q24.insert(0, f'{round(self.output_wages1[3], 3)}')
            self.visualization_q25.delete(0, "end")
            self.visualization_q25.insert(0, f'{round(self.output_wages1[4], 3)}')
            self.visualization_q26.delete(0, "end")
            self.visualization_q26.insert(0, f'{round(self.output_wages1[5], 3)}')
            self.visualization_q27.delete(0, "end")
            self.visualization_q27.insert(0, f'{round(self.output_wages1[6], 3)}')
            self.visualization_q28.delete(0, "end")
            self.visualization_q28.insert(0, f'{round(self.output_wages1[7], 3)}')
            self.visualization_q29.delete(0, "end")
            self.visualization_q29.insert(0, f'{round(self.output_wages1[8], 3)}')

            for i in range(0, len(self.output_wages1)-1):
                self.weights.append(self.output_wages1[i])

            scores3 = NB.evaluate_algorithm(self, self.train_dataset3, NB.back_propagation, n_folds, l_rate, n_epoch, n_hidden, rand_range=0.1, momentum=0, bias=0)
            self.visualization_q31.delete(0, "end")
            self.visualization_q31.insert(0, f'{round(self.output_wages1[0], 3)}')
            self.visualization_q32.delete(0, "end")
            self.visualization_q32.insert(0, f'{round(self.output_wages1[1], 3)}')
            self.visualization_q33.delete(0, "end")
            self.visualization_q33.insert(0, f'{round(self.output_wages1[2], 3)}')
            self.visualization_q34.delete(0, "end")
            self.visualization_q34.insert(0, f'{round(self.output_wages1[3], 3)}')
            self.visualization_q35.delete(0, "end")
            self.visualization_q35.insert(0, f'{round(self.output_wages1[4], 3)}')
            self.visualization_q36.delete(0, "end")
            self.visualization_q36.insert(0, f'{round(self.output_wages1[5], 3)}')
            self.visualization_q37.delete(0, "end")
            self.visualization_q37.insert(0, f'{round(self.output_wages1[6], 3)}')
            self.visualization_q38.delete(0, "end")
            self.visualization_q38.insert(0, f'{round(self.output_wages1[7], 3)}')
            self.visualization_q39.delete(0, "end")
            self.visualization_q39.insert(0, f'{round(self.output_wages1[8], 3)}')

            for i in range(0, len(self.output_wages1)-1):
                self.weights.append(self.output_wages1[i])

            self.__visualization()

            # ##############
            # # 1_EVALUATE #
            # ##############
            # self.scores = list()
            # for self.fold in self.folds:
            #     self.train_set = list(self.folds)
            #     self.train_set = sum(self.train_set, [])
            #     self.test_set = list()
            #     for row in self.fold:
            #         row_copy = list(row)
            #         self.test_set.append(row_copy)
            #         row_copy[-1] = None
            #
            # #self.predicted = NN.back_propagation(self)
            # ##############
            # # 2_BACK PRO #
            # ##############
            #
            #
            # #NN.train_network(self)
            # ##############
            # # 3_TRAINING #
            # ##############
            # print(self.n_epoch)
            # for epoch in range(self.n_epoch):
            #     for self.row_train in self.train_dataset:
            #         #self.outputs = forward_propagate(self)
            #         ##############
            #         # 4_FORW.PRO #
            #         ##############
            #         self.inputs = self.row_train
            #         for self.layer1 in self.network1:
            #             self.new_inputs1 = []
            #             for self.neuron1 in self.layer1:
            #                 self.activate_weights = self.neuron1['weights']
            #                 self.activation = NN.activate(self)
            #                 self.neuron1['output'] = NN.transfer(self)
            #                 self.new_inputs1.append(self.neuron1['output'])
            #             self.inputs1 = self.new_inputs1
            #         self.outputs = self.inputs1
            #         # print('self.outputs:') #TU jest jeszcze OK
            #         # print(self.outputs)
            #         ##############
            #         # 4_FORW.PRO #
            #         ##############
            #         self.expected = [0 for i in range(self.n_outputs)]
            #         self.expected[row[-1]] = 1
            #         #NN.backward_propagate_error(self)
            #         ##############
            #         # 7_BACK_ERR #
            #         ##############
            #         for i in reversed(range(len(self.network1))):
            #             layer = self.network1[i]
            #             errors = list()
            #             if i != len(self.network1) - 1:
            #                 for j in range(len(layer)):
            #                     error = 0.0
            #                     for neuron in self.network1[i + 1]:
            #                         #print(neuron['delta'])
            #                         error += (neuron['weights'][j] * neuron['delta'])
            #                         #print(error) Do poweszenia
            #                     errors.append(error)
            #             else:
            #                 for j in range(len(layer)):
            #                     neuron = layer[j]
            #                     errors.append(neuron['output'] - self.expected[j])
            #                 #print(errors) #w Gui errors są o rzad wielkosci mniejsze niz w NewBP (z wyjatkiem pierwszego)
            #             for j in range(len(layer)):
            #                 neuron = layer[j]
            #                 neuron['delta'] = errors[j] * NN.transfer_derivative(neuron['output'])
            #                 #print(neuron['delta']) roznica rzedu wielkosci
            #         ##############
            #         # 7_BACK_ERR #
            #         ##############
            #         NN.update_weights(self)
            # for i in range(len(self.network1)):
            #     for self.neuron in self.network1[i]:
            #         print("Actual weights:")
            #         print(self.neuron['weights'])
            # ##############
            # # 3_TRAINING #
            # ##############
            #
            # self.predictions = list()
            # print(self.test_dataset)
            #
            # for row_BP in self.test_dataset:
            #     #self.prediction = NN.predict(self.network1, self.row_BP)
            #     ##############
            #     # 5_PREDICT  #
            #     ##############
            #     ##############
            #     # 4_FORW.PRO #
            #     ##############
            #     self.inputs = self.row_train
            #     for self.layer1 in self.network1:
            #         self.new_inputs1 = []
            #         for self.neuron1 in self.layer1:
            #             self.activate_weights = self.neuron1['weights']
            #             self.activation = NN.activate(self)
            #             self.neuron1['output'] = NN.transfer(self)
            #             self.new_inputs1.append(self.neuron1['output'])
            #         self.inputs1 = self.new_inputs1
            #     self.outputs = self.inputs1
            #     self.prediction=self.outputs.index(max(self.outputs))
            #     ##############
            #     # 4_FORW.PRO #
            #     ##############
            #     ##############
            #     # 5_PREDICT  #
            #     ##############
            #
            #     self.predictions.append(self.prediction)
            # return (self.predictions)
            # ##############
            # # 2_BACK PRO #
            # ##############
            # self.actual = [self.row_actual[-1] for self.row_actual in fold]
            # self.accuracy = NN.accuracy_metric(self)
            # self.scores.append(self.accuracy)
            # print(scores)
            # ##############
            # # 1_EVALUATE #
            # ##############
            #
            # #self.scores1 = NN.evaluate_algorithm(self)

    def __visualization(self):
        GOLD = "#ffff00"
        GREY = "#808080"
        BLACK = "#000000"

        coords = [[0, 0],[52, 0],[104, 0],[0, 52],[52, 52],[104, 52],[0, 104],[52, 104],[104, 104]]
        offset = [0, 180, 360]
        width = 50
        i = 0

        #kwadraty
        for of in offset:
            for co in coords:
                self.c.create_rectangle(co[0], co[1]+of, co[0]+width, co[1]+width+of, outline="#000000", fill=BLACK)

                if self.weights[i] > 0:
                    COLOR = GREY
                else:
                    COLOR = GOLD

                if -1 <= self.weights[i] <= 1:
                    self.c.create_rectangle(co[0], co[1]+of, co[0]+abs(self.weights[i])*width, co[1]+abs(self.weights[i])*width+of, outline="#000000", fill=COLOR)
                else:
                    self.c.create_rectangle(co[0], co[1]+of, co[0]+width, co[1]+width+of, outline="#000000", fill=COLOR)

                i += 1
