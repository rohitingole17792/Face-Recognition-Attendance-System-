from tkinter import*
from tkinter import ttk
from PIL import Image,ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
import os
import numpy as np

class Train:
    def __init__(self, root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("face Recogniton System")


        title_lbl=Label(self.root,text="TRAIN DATA SET",font=("times new roman",35,"bold"),bg="black",fg="#6897BB")
        title_lbl.place(x=0,y=0, width=1530,height=45)

        img_top=Image.open(r"collage_img\5.jpg")
        img_top=img_top.resize((1530,325),Image.ANTIALIAS)
        self.photoimg_top=ImageTk.PhotoImage(img_top)


        f_img=Label(self.root,image=self.photoimg_top)
        f_img.place(x=0,y=55,width=1530,height=325)

        #===============Btn==================
        b1_1=Button(self.root, text="Train Data", compound=self.train_classifier(), cursor="hand2", font=("times new roman",25,"bold"), bg="white", fg="pink")
        b1_1.place(x=0,y=380, width=1530,height=60)

        #=================================================
        img_bottom=Image.open(r"collage_img\8.jpg")
        img_bottom=img_bottom.resize((1530,325),Image.ANTIALIAS)
        self.photoimg_bottom=ImageTk.PhotoImage(img_bottom)


        f_img=Label(self.root,image=self.photoimg_bottom)
        f_img.place(x=0,y=440,width=1530,height=325)


        #==================================def function=====================
    def train_classifier(self):
        data_dir=("data")
        path=[os.path.join(data_dir,file) for file in os.listdir(data_dir)]

        faces=[]
        ids=[]


        for image in path:
            img=Image.open(image).convert('L')  #Gray scale img
            imageNp=np.array(img,'uint8')
            id=int(os.path.split(image)[1].split('.')[1])

            faces.append(imageNp)
            ids.append(id)
            cv2.imshow("Traning",imageNp)
            cv2.waitKey(1)==13
        ids=np.array(ids)

        recognizer=cv2.face.LBPHFaceRecognizer_create() 
        recognizer.train(faces,ids)
        recognizer.write("classify.xml") 
        cv2.destroyAllWindows()
        messagebox.showinfo("Result","Traning Data set is Compleated!!!!!")


        

    


if __name__ == "__main__":
    root=Tk()
    obj=Train(root)
    root.mainloop()
