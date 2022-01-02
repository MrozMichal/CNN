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
        self.frame.title('BP gui')
        screen_width = self.frame.winfo_screenwidth()
        screen_height = self.frame.winfo_screenheight()
        self.frame.geometry('1170x660+%d+%d' % ((screen_width - 1170) / 2, (screen_height - 660) / 2))
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
        c.pack()

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
        c.create_rectangle(  0,   0+200,  50,  50+200, outline="#000000", fill="#000000")
        c.create_rectangle( 52,   0+200, 102,  50+200, outline="#000000", fill="#000000")
        c.create_rectangle(104,   0+200, 154,  50+200, outline="#000000", fill="#000000")

        c.create_rectangle(  0,  52+200,  50, 102+200, outline="#000000", fill="#000000")
        c.create_rectangle( 52,  52+200, 102, 102+200, outline="#000000", fill="#000000")
        c.create_rectangle(104,  52+200, 154, 102+200, outline="#000000", fill="#000000")

        c.create_rectangle(  0, 104+200,  50, 154+200, outline="#000000", fill="#000000")
        c.create_rectangle( 52, 104+200, 102, 154+200, outline="#000000", fill="#000000")
        c.create_rectangle(104, 104+200, 154, 154+200, outline="#000000", fill="#000000")

        # III kwadrat
        c.create_rectangle(  0,   0+400,  50,  50+400, outline="#000000", fill="#000000")
        c.create_rectangle( 52,   0+400, 102,  50+400, outline="#000000", fill="#000000")
        c.create_rectangle(104,   0+400, 154,  50+400, outline="#000000", fill="#000000")

        c.create_rectangle(  0,  52+400,  50, 102+400, outline="#000000", fill="#000000")
        c.create_rectangle( 52,  52+400, 102, 102+400, outline="#000000", fill="#000000")
        c.create_rectangle(104,  52+400, 154, 102+400, outline="#000000", fill="#000000")

        c.create_rectangle(  0, 104+400,  50, 154+400, outline="#000000", fill="#000000")
        c.create_rectangle( 52, 104+400, 102, 154+400, outline="#000000", fill="#000000")
        c.create_rectangle(104, 104+400, 154, 154+400, outline="#000000", fill="#000000")

        #c.create_rectangle( 30, 30,  130, 130, outline="#ffff00", fill="#ffff00")   # żółty
        #c.create_rectangle(150, 30, 250, 130, outline="#ffffff", fill="#ffffff")    # biały
        #c.create_rectangle(30, 150, 130, 250, outline="#808080", fill="#808080")    # szary
        #c.create_rectangle(150, 150, 250, 250, outline="#000000", fill="#000000")   # czarny

        # IV kwadrat
        c.create_rectangle(  0+200,   0,  50+200,  50, outline="#ffffff", fill="#ffffff")
        c.create_rectangle( 52+200,   0, 102+200,  50, outline="#ffffff", fill="#ffffff")
        c.create_rectangle(104+200,   0, 154+200,  50, outline="#ffffff", fill="#ffffff")

        c.create_rectangle(  0+200,  52,  50+200, 102, outline="#ffffff", fill="#ffffff")
        c.create_rectangle( 52+200,  52, 102+200, 102, outline="#ffffff", fill="#ffffff")
        c.create_rectangle(104+200,  52, 154+200, 102, outline="#ffffff", fill="#ffffff")

        c.create_rectangle(  0+200, 104,  50+200, 154, outline="#ffffff", fill="#ffffff")
        c.create_rectangle( 52+200, 104, 102+200, 154, outline="#ffffff", fill="#ffffff")
        c.create_rectangle(104+200, 104, 154+200, 154, outline="#ffffff", fill="#ffffff")

        # V kwadrat
        c.create_rectangle(  0+200,   0+200,  50+200,  50+200, outline="#ffffff", fill="#ffffff")
        c.create_rectangle( 52+200,   0+200, 102+200,  50+200, outline="#ffffff", fill="#ffffff")
        c.create_rectangle(104+200,   0+200, 154+200,  50+200, outline="#ffffff", fill="#ffffff")

        c.create_rectangle(  0+200,  52+200,  50+200, 102+200, outline="#ffffff", fill="#ffffff")
        c.create_rectangle( 52+200,  52+200, 102+200, 102+200, outline="#ffffff", fill="#ffffff")
        c.create_rectangle(104+200,  52+200, 154+200, 102+200, outline="#ffffff", fill="#ffffff")

        c.create_rectangle(  0+200, 104+200,  50+200, 154+200, outline="#ffffff", fill="#ffffff")
        c.create_rectangle( 52+200, 104+200, 102+200, 154+200, outline="#ffffff", fill="#ffffff")
        c.create_rectangle(104+200, 104+200, 154+200, 154+200, outline="#ffffff", fill="#ffffff")

        # VI kwadrat
        c.create_rectangle(  0+200,   0+400,  50+200,  50+400, outline="#ffffff", fill="#ffffff")
        c.create_rectangle( 52+200,   0+400, 102+200,  50+400, outline="#ffffff", fill="#ffffff")
        c.create_rectangle(104+200,   0+400, 154+200,  50+400, outline="#ffffff", fill="#ffffff")

        c.create_rectangle(  0+200,  52+400,  50+200, 102+400, outline="#ffffff", fill="#ffffff")
        c.create_rectangle( 52+200,  52+400, 102+200, 102+400, outline="#ffffff", fill="#ffffff")
        c.create_rectangle(104+200,  52+400, 154+200, 102+400, outline="#ffffff", fill="#ffffff")

        c.create_rectangle(  0+200, 104+400,  50, 154+400, outline="#ffffff", fill="#ffffff")
        c.create_rectangle( 52+200, 104+400, 102, 154+400, outline="#ffffff", fill="#ffffff")
        c.create_rectangle(104+200, 104+400, 154, 154+400, outline="#ffffff", fill="#ffffff")


    def __create_visualization_options_widgets(self):
        self.visualization_options_frame = LabelFrame(self.frame, text='Opcje wyświetlania', width=500, height=70)
        self.visualization_options_frame.place(x=620, y=580)

        self.number_of_actualization_label = Label(self.visualization_options_frame, text='Częstość aktualizacji wag:')
        self.number_of_actualization_label.place(x=55, y=15, height=25)
        self.number_of_actualization_var = StringVar()
        self.number_of_actualization_entry = Entry(self.visualization_options_frame, justify=CENTER,textvariable=self.number_of_actualization_var)
        self.number_of_actualization_entry.insert(0, '10')
        self.number_of_actualization_entry.place(x=250, y=15, width=70, height=25)

if __name__ == '__main__':
    gui = Gui()