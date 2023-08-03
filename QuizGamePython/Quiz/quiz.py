import requests
import random
if __name__ == "__main__":
    response = requests.get('https://opentdb.com/api.php?amount=10&difficulty=medium&type=multiple')
    x=1
    score=0
    for i in response.json()['results']:
        print(str(x)+". "+i['question'])
        array=i['correct_answer'].split(",")+i['incorrect_answers']
        random.shuffle(array)
        print("A. "+array[0])
        print("B. "+array[1])
        print("C. "+array[2])
        print("D. "+array[3])
        if array[0]==i['correct_answer']:
            correct='A'
        elif array[1]==i['correct_answer']:
            correct='B'
        elif array[2]==i['correct_answer']:
            correct='C'
        elif array[3]==i['correct_answer']:
            correct='D'
        x+=1
        inp=input("Enter your answer: ")
        if inp==correct:
            score+=1
            print("Correct")
        else:
            print("Incorrect")
        print("---------------------------------")
    print("Game Over : Score = ",score)