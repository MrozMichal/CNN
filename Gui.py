from tkinter import *
from tkinter import messagebox
from NeuralNetwork import NeuralNetwork
from matplotlib import pyplot as plt
import numpy as np

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

        self.__create_widgets()
        self.network = None
        self.training_window = None
        self.frame.mainloop()

    def __create_widgets(self):
        self.__create_data_panel_widgets()
        self.__create_standard_parameters_widgets()
        self.__create_optional_parameters_widgets()
        self.__create_action_panel_widgets()
        self.__create_visualization_widgets()
        self.__create_visualization_options_widgets()

        self.response_surface_button = Button(self.frame, text='Ikona', bg='#E1E1E1',
                                              command=lambda: self.__plot_response_surface())
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

    def __create_data_panel_widgets(self):
        self.data_panel_frame = LabelFrame(self.frame, text='Panel danych', width=500, height=100)
        self.data_panel_frame.place(x=45, y=15)

        self.data_panel_train_button = Button(self.data_panel_frame, text='Wczytaj plik uczący', bg='#E1E1E1',
                                              command=lambda: self.__plot_response_surface())
        self.data_panel_train_button.place(x=15, y=10, width=120, height=25)
        self.data_panel_train_var = StringVar()
        self.data_panel_train_entry = Entry(self.data_panel_frame, justify=CENTER,
                                               textvariable=self.data_panel_train_var,
                                               state='readonly', readonlybackground='#FFFFFF')
        self.data_panel_train_entry.place(x=155, y=10, width=290, height=25)
        self.radio_button_train = Radiobutton(self.data_panel_frame, variable=self.data_panel_train_var, value=0)
        self.radio_button_train.place(x=450, y=10)

        self.data_panel_test_button = Button(self.data_panel_frame, text='Wczytaj plik testowy', bg='#E1E1E1',
                                              command=lambda: self.__plot_response_surface())
        self.data_panel_test_button.place(x=15, y=45, width=120, height=25)
        self.data_panel_test_var = StringVar()
        self.data_panel_test_entry = Entry(self.data_panel_frame, justify=CENTER,
                                               textvariable=self.data_panel_test_var,
                                               state='readonly', readonlybackground='#FFFFFF')
        self.data_panel_test_entry.place(x=155, y=45, width=290, height=25)
        self.radio_button_test = Radiobutton(self.data_panel_frame, variable=self.data_panel_test_var, value=0)
        self.radio_button_test.place(x=450, y=45)

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

        self.action_panel_initialize_button = Button(self.optional_parameters_frame, text='Inicjalizuj sieć', bg='#E1E1E1', command=lambda: self.__plot_response_surface())
        self.action_panel_initialize_button.place(x=15, y=15, width=90, height=40)
        self.action_panel_initialize_var = StringVar()
        self.radio_button_initialize = Radiobutton(self.optional_parameters_frame, variable=self.action_panel_initialize_var, value=0)
        self.radio_button_initialize.place(x=115, y=25)

        self.action_panel_learn_button = Button(self.optional_parameters_frame, text='Naucz sieć', bg='#E1E1E1', command=lambda: self.__plot_response_surface())
        self.action_panel_learn_button.place(x=200, y=15, width=90, height=40)

        self.action_panel_reset_button = Button(self.optional_parameters_frame, text='RESET', bg='#E1E1E1', command=lambda: self.__plot_response_surface())
        self.action_panel_reset_button.place(x=390, y=15, width=90, height=40)

    def __create_visualization_widgets(self):
        self.visualization_frame = LabelFrame(self.frame, text='Wizualizacja',width=500, height=550)
        #self.visualization_frame = LabelFrame(self.frame, text='Wizualizacja',width=500, height=850)
        self.visualization_frame.place(x=620, y=15,width=500, height=550)

        c = Canvas(self.visualization_frame)
        c.pack(fill=BOTH, expand=1)
        #outline="#ffff00", fill="#ffff00")   # żółty
        #outline="#ffffff", fill="#ffffff")   # biały
        #outline="#808080", fill="#808080")   # szary
        #outline="#000000", fill="#000000")   # czarny

        # I kwadrat
        c.create_rectangle(  0,   0,  50,  50, outline="#000000", fill="#000000")
        c.create_rectangle( 52,   0, 102,  50, outline="#000000", fill="#000000")
        c.create_rectangle(104,   0, 154,  50, outline="#000000", fill="#000000")

        c.create_rectangle(  0,  52,  50, 102, outline="#000000", fill="#000000")
        c.create_rectangle( 52,  52, 102, 102, outline="#000000", fill="#000000")
        c.create_rectangle(104,  52, 154, 102, outline="#000000", fill="#000000")

        c.create_rectangle(  0, 104,  50, 154, outline="#000000", fill="#000000")
        c.create_rectangle( 52, 104, 102, 154, outline="#000000", fill="#000000")
        c.create_rectangle(104, 104, 154, 154, outline="#000000", fill="#000000")

        # II kwadrat
        c.create_rectangle(  0,   0+180,  50,  50+180, outline="#000000", fill="#000000")
        c.create_rectangle( 52,   0+180, 102,  50+180, outline="#000000", fill="#000000")
        c.create_rectangle(104,   0+180, 154,  50+180, outline="#000000", fill="#000000")

        c.create_rectangle(  0,  52+180,  50, 102+180, outline="#000000", fill="#000000")
        c.create_rectangle( 52,  52+180, 102, 102+180, outline="#000000", fill="#000000")
        c.create_rectangle(104,  52+180, 154, 102+180, outline="#000000", fill="#000000")

        c.create_rectangle(  0, 104+180,  50, 154+180, outline="#000000", fill="#000000")
        c.create_rectangle( 52, 104+180, 102, 154+180, outline="#000000", fill="#000000")
        c.create_rectangle(104, 104+180, 154, 154+180, outline="#000000", fill="#000000")

        # III kwadrat
        c.create_rectangle(  0,   0+360,  50,  50+360, outline="#000000", fill="#000000")
        c.create_rectangle( 52,   0+360, 102,  50+360, outline="#000000", fill="#000000")
        c.create_rectangle(104,   0+360, 154,  50+360, outline="#000000", fill="#000000")

        c.create_rectangle(  0,  52+360,  50, 102+360, outline="#000000", fill="#000000")
        c.create_rectangle( 52,  52+360, 102, 102+360, outline="#000000", fill="#000000")
        c.create_rectangle(104,  52+360, 154, 102+360, outline="#000000", fill="#000000")

        c.create_rectangle(  0, 104+360,  50, 154+360, outline="#000000", fill="#000000")
        c.create_rectangle( 52, 104+360, 102, 154+360, outline="#000000", fill="#000000")
        c.create_rectangle(104, 104+360, 154, 154+360, outline="#000000", fill="#000000")

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
                                           textvariable=self.visualization_q11,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q12 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q12,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q13 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q13,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q14 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q14,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q15 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q15,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q16 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q16,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q17 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q17,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q18 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q18,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q19 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q19,
                                           state='readonly', readonlybackground='#FFFFFF')

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
                                           textvariable=self.visualization_q21,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q22 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q22,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q23 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q23,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q24 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q24,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q25 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q25,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q26 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q26,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q27 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q27,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q28 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q28,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q29 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q29,
                                           state='readonly', readonlybackground='#FFFFFF')

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
                                           textvariable=self.visualization_q31,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q32 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q32,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q33 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q33,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q34 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q34,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q35 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q35,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q36 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q36,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q37 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q37,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q38 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q38,
                                           state='readonly', readonlybackground='#FFFFFF')
        self.visualization_q39 = Entry(self.visualization_frame, justify=CENTER,
                                           textvariable=self.visualization_q39,
                                           state='readonly', readonlybackground='#FFFFFF')

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

        c.pack(fill=BOTH, expand=1)

    def __create_visualization_options_widgets(self):
        self.visualization_options_frame = LabelFrame(self.frame, text='Opcje wyświetlania', width=500, height=70)
        self.visualization_options_frame.place(x=620, y=580)

        self.number_of_actualization_label = Label(self.visualization_options_frame, text='Częstość aktualizacji wag:')
        self.number_of_actualization_label.place(x=55, y=15, height=25)
        self.number_of_actualization_var = StringVar()
        self.number_of_actualization_entry = Entry(self.visualization_options_frame, justify=CENTER,textvariable=self.number_of_actualization_var)
        self.number_of_actualization_entry.insert(0, '10')
        self.number_of_actualization_entry.place(x=250, y=15, width=70, height=25)
