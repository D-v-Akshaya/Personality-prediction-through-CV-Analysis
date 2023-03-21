import os
from tkinter import *
from tkinter.font import Font
from tkinter import filedialog
from PIL import ImageTk,Image
import mPreditPersonality as p
from functools import partial


def page3(name,gen,age,openesss,neuro,consci,agree,extro):
    root3=Tk()
    root3.geometry('700x500')
    root3.configure(background='#a9a0d8')
    root3.title("Personality")
    lab=Label(root3, text="Personality Prediction", foreground='black', font=('Helvetica', 20, 'bold'), pady=10).pack()
    l1=Label(root3, text="Name: "+name, foreground='white', bg='black').place(x=70, y=130)
    l2=Label(root3, text="Age: "+str(age), foreground='white', bg='black').place(x=70, y=160)
    
def checkValue(name,ag,gen,open,neu,con,agr,ext):
    try:
        age=int(ag)
        openesss=int(open)
        neuro=int(neu)
        consci=int(con)
        agree=int(agr)
        extro=int(ext)
    except:
        page2()
    else:
        page3(name,gen,age,openesss,neuro,consci,agree,extro)

def page2():
    i=0
    root.withdraw()
    # Creating new window
    top = Tk()
    top.geometry('700x500')
    top.configure(background='lightblue')
    top.title("Apply For A Job")
    
    #Title
    lab=Label(top, text="Personality Prediction", foreground='black', font=('Helvetica', 20, 'bold'), pady=10).pack()

    #Job_Form
    job_list=('Select Job', '101-Developer at TTC', '102-Chef at Taj', '103-Professor at MIT')
    job = StringVar(top)
    job.set(job_list[0])

    l1=Label(top, text="Applicant Name", foreground='white', bg='black').place(x=70, y=130)
    l2=Label(top, text="Age", foreground='white', bg='black').place(x=70, y=160)
    l3=Label(top, text="Gender", foreground='white', bg='black').place(x=70, y=190)
    l4=Label(top, text="Upload Resume", foreground='white', bg='black').place(x=70, y=220)
    l5=Label(top, text="Enjoy New Experience or thing(Openness)", foreground='white', bg='black').place(x=70, y=250)
    l6=Label(top, text="How Offen You Feel Negativity(Neuroticism)", foreground='white', bg='black').place(x=70, y=280)
    l7=Label(top, text="Wishing to do one's work well and thoroughly(Conscientiousness)", foreground='white', bg='black').place(x=70, y=310)
    l8=Label(top, text="How much would you like work with your peers(Agreeableness)", foreground='white', bg='black').place(x=70, y=340)
    l9=Label(top, text="How outgoing and social interaction you like(Extraversion)", foreground='white', bg='black').place(x=70, y=370)
    
    sName=Entry(top)
    sName.place(x=460, y=130, width=160)
    age=Entry(top)
    age.place(x=460, y=160, width=160)
    gender = IntVar()
    R1 = Radiobutton(top, text="Male", variable=gender, value=1, padx=7)
    R1.place(x=460, y=190)
    R2 = Radiobutton(top, text="Female", variable=gender, value=2, padx=3)
    R2.place(x=550, y=190)
    cv=Button(top, text="Select File", command=lambda:  OpenFile(cv))
    cv.place(x=460, y=220, width=160)
    openness=Entry(top)
    openness.insert(0,'1-10')
    openness.place(x=460, y=250, width=160)
    neuroticism=Entry(top)
    neuroticism.insert(0,'1-10')
    neuroticism.place(x=460, y=280, width=160)
    conscientiousness=Entry(top)
    conscientiousness.insert(0,'1-10')
    conscientiousness.place(x=460, y=310, width=160)
    agreeableness=Entry(top)
    agreeableness.insert(0,'1-10')
    agreeableness.place(x=460, y=340, width=160)
    extraversion=Entry(top)
    extraversion.insert(0,'1-10')
    extraversion.place(x=460, y=370, width=160)
    submit1=Button(top, padx=2, pady=0,text="Submit",command=lambda :checkValue( name=str(sName.get()),ag=age.get(),gen=int(gender.get()),open=openness.get(),neu=neuroticism.get(),con=conscientiousness.get(),agr=agreeableness.get(),ext=extraversion.get()),font=('bold'))
    submit1.place(x=350, y=400, width=200)
    top.mainloop() 
       
        


def OpenFile(b4):
    global loc;
    name = filedialog.askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
                            filetypes =(("Document","*.docx*"),("PDF","*.pdf*"),('All files', '*')),
                           title = "Choose a file."
                           )
    try:
        filename=os.path.basename(name)
        loc=name
    except:
        filename=name
        loc=name
    b4.config(text=filename)
    return

root= Tk()
root.title("Home page")
root['bg']='#ffe2e6'
root.geometry('700x500')
#f= ('Times roman',20)
#page1
h1=Label(root, text="\t\t Welcome to Personality prediction through CV Analysis !!!!!\t\t\t", font=('time roman',25),bg='#ffc0cb')
h1.pack()
f0=Frame(root,bg='#ffe2e6')
f0.pack()
next1=Button(f0,text="Predict Personality",command=page2,font=('bold',20))
next1.grid(row=14,column=2,pady=20)
Button(f0,text='Exit',command=root.destroy,font=('bold',20)).grid(row=15,column=2)
root.mainloop()


