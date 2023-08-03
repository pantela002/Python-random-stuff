import tkinter as tk
import requests
from PIL import ImageTk, Image
import random
class Game:
    def __init__(self) -> None:
        self.root=tk.Tk()
        self.root.geometry("1200x563")
        self.root.title("My GUI Program")
        self.root.resizable(False, False)
        #define static variables
        self.canvas = tk.Canvas(self.root, width=1200, height=563)
        self.canvas.pack()
        self.score = 0
        self.response = requests.get('https://opentdb.com/api.php?amount=10&category=31&difficulty=easy&type=multiple')
        self.question_number = 0
        self.run()

    def set_buttons(self,correct_answer,incorrect_answers):

        array=correct_answer+incorrect_answers
        random.shuffle(array)
        btn1 = tk.Button(self.canvas, text=array[0], font=("Courier New",16),activebackground='#24ffe9', activeforeground='white', bg='white', fg='#1760ff', width=35, height=2, command = lambda : self.checkAnswer(btn1,correct_answer))
        btn2 = tk.Button(self.canvas, text=array[1], font=("Courier New",16),activebackground='#24ffe9', activeforeground='white', bg='white', fg='#1760ff', width=35, height=2, command = lambda : self.checkAnswer(btn2,correct_answer))
        btn3 = tk.Button(self.canvas, text=array[2], font=("Courier New",16),activebackground='#24ffe9', activeforeground='white', bg='white', fg='#1760ff', width=35, height=2, command = lambda : self.checkAnswer(btn3,correct_answer))
        btn4 = tk.Button(self.canvas, text=array[3], font=("Courier New",16),activebackground='#24ffe9', activeforeground='white', bg='white', fg='#1760ff', width=35, height=2, command = lambda : self.checkAnswer(btn4,correct_answer))
        btn1.place(relx='.165', rely='.5')
        btn2.place(relx='.52', rely='.5')
        btn3.place(relx='.165', rely='.65')
        btn4.place(relx='.52', rely='.65')

    def checkAnswer(self,button,correct_answer):
        if button['text'] == correct_answer[0]:
            #print(button['text'])
            self.score += 1
            button['activebackground'] = '#0cf232'
            button['bg']='#0cf232'
            button['text'] = 'Correct'
        else:
            button['activebackground'] = '#ff2ec7'
            button['bg'] = '#ff2ec7'
            button['text'] = 'Incorrect'
        self.canvas.after(1000,self.reset_text)
    
    def reset_text(self):
        self.question_number += 1
        self.canvas.delete("all")
        self.set_question()


    def set_question(self):
        self.img = ImageTk.PhotoImage(Image.open("2.png"))
        self.canvas.create_image(0, 0, image=self.img, anchor='nw')
        if self.question_number == 10:
            print("Game Over : Score = ",self.score)
            exit() 
        question = self.response.json()['results'][self.question_number]['question']
        correct_answer = self.response.json()['results'][self.question_number]['correct_answer'].split(",")
        incorrect_answers = self.response.json()['results'][self.question_number]['incorrect_answers']
        question_length = len(question)
        if question_length <= 100:
            font_size = 20
        else:
            font_size = 16

        questionlabel = tk.Label(self.canvas,width=60, height=3 , text=question, font=("Courier New",font_size,'bold'), bg='white', fg='#1760ff', wraplength=800, justify="center", compound="center")
        questionlabel.place(relx='.5', rely='.34',anchor='center')
        self.canvas.create_text(900, 50, text="Score: "+str(self.score), font=("Ariel ", 36, "bold"), fill="#000000")
        self.set_buttons(correct_answer,incorrect_answers)
        self.set_score()
        self.root.update() 

    def set_score(self):
        canvas1 = tk.Canvas(self.root, width=1200, height=20,bg="#ffffff")
        canvas1.place(relx='.5', rely='.0',anchor='center')
        # calculate the coordinates of the two rectangles
        total_width = 1200
        colored_width = total_width * self.question_number / 10


        # create the first rectangle for the colored portion
        canvas1.create_rectangle(0, 0, colored_width, 20, fill="#0cf232")

        # create the second rectangle for the remaining portion
        canvas1.create_rectangle(colored_width, 0, total_width, 20, fill="#ffffff")

    def run(self):
        self.set_question()
        self.root.mainloop()
        

if __name__ == "__main__":
    game = Game()