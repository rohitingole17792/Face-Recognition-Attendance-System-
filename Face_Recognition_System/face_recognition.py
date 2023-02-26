from tkinter import*
from tkinter import ttk
from PIL import Image,ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
import os
import numpy as np

class Face_Recognition:
    def __init__(self, root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("face Recogniton System")

        title_lbl=Label(self.root,text="Face Recognition",font=("times new roman",35,"bold"),bg="#FCDFEE",fg="#6897BB")
        title_lbl.place(x=0,y=0, width=1530,height=60)
        #1st img=============
        img9_top=Image.open(r"collage_img\10.jpg")
        img9_top=img9_top.resize((650,700),Image.ANTIALIAS)
        self.photoimg9_top=ImageTk.PhotoImage(img9_top)


        fa_img=Label(self.root,image=self.photoimg9_top)
        fa_img.place(x=0,y=55,width=650,height=700)

        #2nd img==============
        img8_bottom=Image.open(r"collage_img\12.jpg")
        img8_bottom=img8_bottom.resize((950,700),Image.ANTIALIAS)
        self.photoimg8_bottom=ImageTk.PhotoImage(img8_bottom)


        fs_img=Label(self.root,image=self.photoimg8_bottom)
        fs_img.place(x=650,y=55,width=950,height=700)

        #3rd img======================
       # img10_bottom=Image.open(r"collage_img\9.jpg")
        #img10_bottom=img10_bottom.resize((950,700),Image.ANTIALIAS)
        #self.photoimg10_bottom=ImageTk.PhotoImage(img10_bottom)


        #fd_img=Label(self.root,image=self.photoimg10_bottom)
        #fd_img.place(x=652,y=100,width=380,height=600)

        #===============Btn==================
        b1_1=Button(self.root,text="Train",command=self.face_recog,cursor="hand2", font=("times new roman",25,"bold"), bg="white", fg="green")
        b1_1.place(x=785,y=559, width=125,height=40)

        #=========(========================================
    def face_recog(self):
        def draw_boundray(img,classifier,scaleFactor,minNeighbours,color,text,clf):
            gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            features=classifier.detectMultiScale(gray_image,scaleFactor,minNeighbours)

            coord=[]

            for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                id,predict=clf.predict(gray_image)[y:y+h,x:x+w]
                confidence=int((100*(1 - predict/300)))

                conn=mysql.connector.connect(host="localhost",username="root",password="Indianrrp@123",database="codewithrohit")
                my_cursor=conn.cursor()

                my_cursor.execute("select Name from student where Student_id="+str(id))
                n=my_cursor.fetchone()
                n="+".join(n)

                my_cursor.execute("select Roll from student where Student_id="+str(id))
                r=my_cursor.fetchone()
                r="+".join(r)

                my_cursor.execute("select Dep from student where Student_id="+str(id))
                d=my_cursor.fetchone()
                d="+".join(d)

                if confidence>77:
                    cv2.putText(img,f"Roll{r}",(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    cv2.putText(img,f"Name{n}",(x,y-30),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    cv2.putText(img,f"Department{d}",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                else:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(img,"Unknown Face",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                coord[x,y,w,y]

            return coord
        def recognize(img,clf,faceCascade):
            coord=draw_boundray(img,faceCascade,1.1,10(255,25,255),"Face",clf)
            return img

        faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf=cv2.face.LBPHFaceRecognizer_create() 
        clf.read("classify.xml")

        video_cap=cv2.VideoCapture(0)

        while True:
            ret,img=video_cap.read()
            img=recognize(img,clf,faceCascade)
            cv2.imshow("Welcome To Face Recognition",img)

            if cv2.waitKey(1)==13:
                break
            video_cap.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    root=Tk()
    obj=Face_Recognition(root)
    root.mainloop()
