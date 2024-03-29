from tkinter import*
from tkinter import ttk
from PIL import Image,ImageTk
from tkinter import messagebox
import mysql.connector

class Student:
    def __init__(self, root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("face Recogniton System")


     #=========================variables============================================#
        self.var_dep=StringVar()
        self.var_course=StringVar()
        self.var_year=StringVar()
        self.var_semester=StringVar()
        self.va_std_id=StringVar()
        self.var_std_name=StringVar()
        self.var_div=StringVar()
        self.var_roll=StringVar()
        self.var_gender=StringVar()
        self.var_dob=StringVar()
        self.var_email=StringVar()
        self.var_phone=StringVar()
        self.var_address=StringVar()
        self.var_teacher=StringVar()
        

        #first img
        img=Image.open(r"C:\Users\rohit\OneDrive\Desktop\Face_Recognition_System\collage_img\swami.jpeg")
        img=img.resize((500,130),Image.ANTIALIAS)
        self.photoimg=ImageTk.PhotoImage(img)

        f_lbl=Label(self.root,image=self.photoimg)
        f_lbl.place(x=0,y=0,width=500,height=130)

        #2nd img
        img1=Image.open(r"C:\Users\rohit\OneDrive\Desktop\Face_Recognition_System\collage_img\7.jpg")
        img1=img1.resize((500,130),Image.ANTIALIAS)
        self.photoimg1=ImageTk.PhotoImage(img1)

        f_lbl=Label(self.root,image=self.photoimg1)
        f_lbl.place(x=500,y=0,width=500,height=130)

        #3rd img
        img2=Image.open(r"C:\Users\rohit\OneDrive\Desktop\Face_Recognition_System\collage_img\logo.jpeg")
        img2=img2.resize((550,130),Image.ANTIALIAS)
        self.photoimg2=ImageTk.PhotoImage(img2)


        f_lbl=Label(self.root,image=self.photoimg2)
        f_lbl.place(x=1000,y=0,width=550,height=130)

        #bg img
        img3=Image.open(r"C:\Users\rohit\OneDrive\Desktop\Face_Recognition_System\collage_img\8.jpg")
        img3=img3.resize((1530,680),Image.ANTIALIAS)
        self.photoimg3=ImageTk.PhotoImage(img3)


        bg_img=Label(self.root,image=self.photoimg3)
        bg_img.place(x=0,y=130,width=1530,height=680)

        title_lbl=Label(bg_img,text="STUDENT MANAGEMENT SYSTEM",font=("times new roman",35,"bold",),bg="black",fg="green")
        title_lbl.place(x=0,y=0, width=1530,height=45)

        main_frame=Frame(bg_img,bd=2)
        main_frame.place(x=0,y=55,width=1520,height=650)

        #===================================================== left label frame =================================================#
        Left_frame=LabelFrame (main_frame,bd=2,bg="white",fg="pink", relief=RIDGE, text="Student Details",font=("times new roman",18,"bold")) 
        Left_frame.place(x=0,y=0,width=730,height=580)

        #-----------------------------------------------------#Current Cource Info#-------------------------------------------------------#
        current_course_frame=LabelFrame (main_frame,bd=2,bg="white", relief=RIDGE, text="Current Course Information ",font=("times new roman",18,"bold")) 
        current_course_frame.place(x=10,y=25,width=710,height=155)

        #Department
        dep_label=Label(current_course_frame,text="Department",font=("times new roman",15,"bold"),bg="white")
        dep_label.grid(row=0,column=0,padx=5,pady=5,sticky=W)

        dep_combo=ttk.Combobox(current_course_frame,textvariable=self.var_dep,font=("times new roman",12,"bold"),state="readonly",width=20)
        dep_combo["values"]=("Select Department","MSc(CS)","MSc(CA)","MSc(CN)","MCA")
        dep_combo.current(0)
        dep_combo.grid(row=0,column=1,padx=2,pady=10,sticky=W)

        #Cource
        cource_label=Label(current_course_frame,text="Cource",font=("times new roman",15,"bold"),bg="white")
        cource_label.grid(row=0,column=2,padx=5,pady=5,sticky=W)

        cource_combo=ttk.Combobox(current_course_frame,textvariable=self.var_course,font=("times new roman",12,"bold"),state="readonly",width=20)
        cource_combo["values"]=("Cource","FY","SE","Deplo")
        cource_combo.current(0)
        cource_combo.grid(row=0,column=3,padx=2,pady=10,sticky=W)

        #Year
        year_label=Label(current_course_frame,text="Year",font=("times new roman",15,"bold"),bg="white")
        year_label.grid(row=1,column=0,padx=5,pady=5,sticky=W)

        year_combo=ttk.Combobox(current_course_frame,textvariable=self.var_year,font=("times new roman",12,"bold"),state="readonly",width=20)
        year_combo["values"]=("Select Year","2020-21","2021-22","2022-23","2023-24")
        year_combo.current(0)
        year_combo.grid(row=1,column=1,padx=2,pady=10,sticky=W)

        #Semster
        cource_label=Label(current_course_frame,text="Semester",font=("times new roman",15,"bold"),bg="white")
        cource_label.grid(row=1,column=2,padx=5,pady=5,sticky=W)

        cource_combo=ttk.Combobox(current_course_frame,textvariable=self.var_semester,font=("times new roman",12,"bold"),state="readonly",width=20)
        cource_combo["values"]=("Select Semester","Sem-1st","Sem-2nd","Sem-3rd","Sem-4th",)
        cource_combo.current(0)
        cource_combo.grid(row=1,column=3,padx=2,pady=10,sticky=W)

        #====================================================#Class Student Info#=========================================================#
        class_Student_frame=LabelFrame (main_frame,bd=2,bg="white", relief=RIDGE, text="Class Student Information ",font=("times new roman",18,"bold")) 
        class_Student_frame.place(x=5,y=180,width=710,height=400)

        #Student Id
        studentID_label=Label(class_Student_frame,text="StudentID:",font=("times new roman",15,"bold"),bg="white")
        studentID_label.grid(row=0,column=0,padx=5,pady=5,sticky=W)

        studentID_entry=ttk.Entry(class_Student_frame,textvariable=self.va_std_id,width=20,font=("times new roman",15,"bold"))
        studentID_entry.grid(row=0,column=1,padx=10,sticky=W)

        #Student name
        studentName_label=Label(class_Student_frame,text="Student Name:",font=("times new roman",15,"bold"),bg="white")
        studentName_label.grid(row=0,column=2,padx=5,pady=5,sticky=W)

        studentName_entry=ttk.Entry(class_Student_frame,textvariable=self.var_std_name,width=20,font=("times new roman",15,"bold"))
        studentName_entry.grid(row=0,column=3,padx=10,sticky=W)

        #Class Division
        Class_div_label=Label(class_Student_frame,text="Class Div:",font=("times new roman",15,"bold"),bg="white")
        Class_div_label.grid(row=1,column=0,padx=5,pady=5,sticky=W)

        Class_div_entry=ttk.Entry(class_Student_frame,textvariable=self.var_div,width=20,font=("times new roman",15,"bold"))
        Class_div_entry.grid(row=1,column=1,padx=10,sticky=W)

        #Roll No
        roll_no_label=Label(class_Student_frame,text="Roll No:",font=("times new roman",15,"bold"),bg="white")
        roll_no_label.grid(row=1,column=2,padx=5,pady=5,sticky=W)

        roll_no_entry=ttk.Entry(class_Student_frame,textvariable=self.var_roll,width=20,font=("times new roman",15,"bold"))
        roll_no_entry.grid(row=1,column=3,padx=10,sticky=W)

        #Gender
        gender_label=Label(class_Student_frame,text="Gender:",font=("times new roman",15,"bold"),bg="white")
        gender_label.grid(row=2,column=0,padx=5,pady=5,sticky=W)

        gender_combo=ttk.Combobox(class_Student_frame,textvariable=self.var_gender,font=("times new roman",12,"bold"),state="readonly",width=20)
        gender_combo["values"]=("Select","Female","Male","Other")
        gender_combo.current(0)
        gender_combo.grid(row=2,column=1,padx=2,pady=10,sticky=W)

        #DOB
        dob_label=Label(class_Student_frame,text="DOB:",font=("times new roman",15,"bold"),bg="white")
        dob_label.grid(row=2,column=2,padx=5,pady=5,sticky=W)

        dob_entry=ttk.Entry(class_Student_frame,textvariable=self.var_dob,width=20,font=("times new roman",15,"bold"))
        dob_entry.grid(row=2,column=3,padx=10,sticky=W)

        #Email
        email_label=Label(class_Student_frame,text="Email:",font=("times new roman",15,"bold"),bg="white")
        email_label.grid(row=3,column=0,padx=5,pady=5,sticky=W)

        email_entry=ttk.Entry(class_Student_frame,textvariable=self.var_email,width=20,font=("times new roman",15,"bold"))
        email_entry.grid(row=3,column=1,padx=10,sticky=W)

        #Cell no
        phone_label=Label(class_Student_frame,text="Phone No:",font=("times new roman",15,"bold"),bg="white")
        phone_label.grid(row=3,column=2,padx=5,pady=5,sticky=W)

        phone_entry=ttk.Entry(class_Student_frame,textvariable=self.var_phone,width=20,font=("times new roman",15,"bold"))
        phone_entry.grid(row=3,column=3,padx=10,sticky=W)

        #Adress
        address_label=Label(class_Student_frame,text="Address:",font=("times new roman",15,"bold"),bg="white")
        address_label.grid(row=4,column=0,padx=5,pady=5,sticky=W)

        address_entry=ttk.Entry(class_Student_frame,textvariable=self.var_address,width=20,font=("times new roman",15,"bold"))
        address_entry.grid(row=4,column=1,padx=10,sticky=W)

        #Teacher name
        teacher_label=Label(class_Student_frame,text="Teacher Name:",font=("times new roman",15,"bold"),bg="white")
        teacher_label.grid(row=4,column=2,padx=5,pady=5,sticky=W)

        teacher_entry=ttk.Entry(class_Student_frame,textvariable=self.var_teacher,width=20,font=("times new roman",15,"bold"))
        teacher_entry.grid(row=4,column=3,padx=10,sticky=W)


        # radio Buttons
        self.var_radio1=StringVar()
        radionbtn1=ttk.Radiobutton(class_Student_frame,variable=self.var_radio1,text="Take Photo Sample", value="Yes")
        radionbtn1.grid(row=6, column=0)

        radionbtn2=ttk. Radiobutton(class_Student_frame,variable=self.var_radio1,text="No Photo Sample", value="No") 
        radionbtn2.grid(row=6, column=1)

        #bbuttons frame
        btn_frame=Frame(class_Student_frame,bd=2,relief=RIDGE,bg="white") 
        btn_frame.place(x=0,y=260,width=710,height=45)

        #save
        save_btn=Button(btn_frame,text="Save",command=self.add_data,width="14",font=("times new roman",15,"bold"),bg="#6897BB")
        save_btn.grid(row=0,column=0)

        #update
        update_btn=Button(btn_frame,text="Update",command=self.update_data,width="14",font=("times new roman",15,"bold"),bg="#6897BB")
        update_btn.grid(row=0,column=1)

        #Delete
        delete_btn=Button(btn_frame,text="Delete",command=self.delete_data,width="14",font=("times new roman",15,"bold"),bg="#6897BB")
        delete_btn.grid(row=0,column=2)

        #Resrt
        reset_btn=Button(btn_frame,text="Reset",width="14",font=("times new roman",15,"bold"),bg="#6897BB")
        reset_btn.grid(row=0,column=3)

        #bbuttons frame1 photo(take & update)
        btn_frame1=Frame(class_Student_frame,bd=2,relief=RIDGE,bg="white") 
        btn_frame1.place(x=0,y=300,width=730,height=45)

        #Take photo
        take_photo_btn=Button(btn_frame1,text="Take Photo Sample",width="30",font=("times new roman",15,"bold"),bg="#6897BB")
        take_photo_btn.grid(row=0,column=0)

        #Update photo
        update_photo_btn=Button(btn_frame1,text="Update Photo Sample",width="30",font=("times new roman",15,"bold"),bg="#6897BB")
        update_photo_btn.grid(row=0,column=1)












        # Right label frame
        Right_frame=LabelFrame (main_frame, bd=2, bg="white",fg="#08403F",relief=RIDGE, text="Student Details",font=("times new roman",18,"bold"))
        Right_frame.place(x=750,y=10,width=730,height=580)

        #============================================Search systm===========================================================#
        search_frame=LabelFrame (Right_frame,bd=2,bg="white",fg="#8D75CF", relief=RIDGE, text="Search System",font=("times new roman",18,"bold")) 
        search_frame.place(x=5,y=10,width=710,height=80)

        search_label=Label(search_frame,text="Search By:",font=("times new roman",13,"bold"),bg="red",fg="white")
        search_label.grid(row=0,column=0,padx=10,pady=5,sticky=W)

        
        search_combo=ttk.Combobox(search_frame,font=("times new roman",10,"bold"),state="readonly",width=20)
        search_combo["values"]=("Select","Roll No","Phone No")
        search_combo.current(0)
        search_combo.grid(row=0,column=1,padx=2,pady=10,sticky=W)

        search_entry=ttk.Entry(search_frame,width=13,font=("times new roman",15,"bold"))
        search_entry.grid(row=0,column=2,padx=10,sticky=W)

        search_btn=Button(search_frame,text="Search",width="10",font=("times new roman",15,"bold"),bg="#6897BB")
        search_btn.grid(row=0,column=3,padx=4)

        showAll_btn=Button(search_frame,text="Show All",width="10",font=("times new roman",15,"bold"),bg="#6897BB")
        showAll_btn.grid(row=0,column=4,padx=4)

        #===================================Table Frame====================================================#

        table_frame=Frame(Right_frame,bd="2",bg="white",relief=RIDGE)
        table_frame.place(x=5,y=100,width=710,height=250)

        scroll_x=ttk.Scrollbar(table_frame,orient=HORIZONTAL)
        scroll_y=ttk.Scrollbar(table_frame,orient=VERTICAL)

        self.student_table=ttk.Treeview(table_frame,column=("dep","course","year","sem","id","name","div","roll","gender","dob","email","phone","address","teacher","photo"),xscrollcommand=scroll_x.set,yscrollcommand=scroll_y.set)

        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.config(command=self.student_table.xview)
        scroll_y.config(command=self.student_table.yview)

        self.student_table.heading("dep", text="Department")
        self.student_table.heading("course", text="Course")
        self.student_table.heading("year", text="Year")
        self.student_table.heading("sem", text="Semester")
        self.student_table.heading("id", text="Student_id") 
        self.student_table.heading("name",text="Name")
        self.student_table.heading("div", text="Division")
        self.student_table.heading("roll", text="Roll No")
        self.student_table.heading("gender", text="Gender")
        self.student_table.heading("dob", text="DOB")
        self.student_table.heading("email", text="Email")
        self.student_table.heading("phone", text="Phone")
        self.student_table.heading("address", text="Address") 
        self.student_table.heading("teacher", text="Teacher")
        self.student_table.heading("photo", text="PhotoSampleStatus")
        self.student_table["show"]="headings"

        self.student_table.column("dep",width=100)
        self.student_table.column("course",width=100)
        self.student_table.column("year",width=100)
        self.student_table.column("sem",width=100)
        self.student_table.column("id",width=100)
        self.student_table.column("name",width=100)
        self.student_table.column("div",width=100)
        self.student_table.column("roll",width=100)
        self.student_table.column("gender",width=100)
        self.student_table.column("dob",width=100)
        self.student_table.column("email",width=100)
        self.student_table.column("phone",width=100)
        self.student_table.column("address",width=100)
        self.student_table.column("teacher",width=100)
        self.student_table.column("photo",width=100)
        

        self.student_table.pack(fill=BOTH, expand=1)
        self.student_table.bind("<ButtonRelease>",self.get_cursor)
        self.fetch_data()

        #=====================funcation decration=============================#
    def add_data(self):
        if self.var_dep.get()=="Select Department" or self.var_std_name.get()=="" or self.va_std_id.get()=="":
            messagebox.showerror("Errors","All filds are required",parent=self.root)
        else:
            try:
                conn=mysql.connector.connect(host="localhost",username="root",password="Indianrrp@123",database="codewithrohit")
                my_cursor=conn.cursor()     
                my_cursor.execute("insert into student values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(
                                                                                                                    self.var_dep.get(),
                                                                                                                    self.var_course.get(),
                                                                                                                    self.var_year.get(),
                                                                                                                    self.var_semester.get(),
                                                                                                                    self.va_std_id.get(),
                                                                                                                    self.var_std_name.get(),
                                                                                                                    self.var_div.get(),
                                                                                                                    self.var_roll.get(),
                                                                                                                    self.var_gender.get(),
                                                                                                                    self.var_dob.get(),
                                                                                                                    self.var_email.get(),
                                                                                                                    self.var_phone.get(),
                                                                                                                    self.var_address.get(),
                                                                                                                    self.var_teacher.get(),
                                                                                                                    self.var_radio1.get()
                                                                                                                ))
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Success","Student details has been added successfully",parent=self.root)                                                                                                  
            except Exception as es:
                messagebox.showerror("Error",f"Due To :{str(es)}",parent=self.root)

                #========================featch data====================================#
    def fetch_data(self):
        conn=mysql.connector.connect(host="localhost",username="root",password="Indianrrp@123",database="codewithrohit")
        my_cursor=conn.cursor()
        my_cursor.execute("select * from student")
        data=my_cursor.fetchall()

        if len(data)!=0:
            self.student_table.delete(*self.student_table.get_children())
            for i in data:
                self.student_table.insert("",END,values=i)
            conn.commit()
        conn.close()
        #==================get cursor================#
    def get_cursor(self,event=""):
        cursor_focus=self.student_table.focus() 
        content=self.student_table.item(cursor_focus)
        data=content ["values"]

       

        self.var_dep.set(data[0]),
        self.var_course.set(data[1]),
        self.var_year.set(data[2]),
        self.var_semester.set(data [3]),
        self.va_std_id.set(data[4]),
        self.var_std_name.set(data[5]),
        self.var_div.set(data[6]),
        self.var_roll.set(data[7]),
        self.var_gender.set(data[8]),
        self.var_dob.set(data[9]),     
        self.var_email.set(data[10]),
        self.var_phone.set(data[11]),
        self.var_address.set(data[12]),
        self.var_teacher.set(data[13]),
        self.var_radio1.set(data[14])

        #===========================update funcation=====================================================#
    def update_data(self):
        if self.var_dep.get()=="Select Department" or self.var_std_name.get()=="" or self.va_std_id.get()=="":
            messagebox.showerror("Errors","All filds are required",parent=self.root)    
        else:
            try:
                Upadate=messagebox.askyesno("Upadte","Do you want to update this student details",parent=self.root)
                if Upadate>0:
                    conn=mysql.connector.connect(host="localhost",username="root",password="Indianrrp@123",database="codewithrohit")
                    my_cursor=conn.cursor()
                    my_cursor.execute("update student set Dep=%s,course=%s,Year=%s,Semester=%s,Name=%s,Division=%s,Roll=%s,Gender=%s,Dob=%s,Email=%s,Phone=%s,Address=%s,Teacher=%s,PhotoSample=%s where Student_id=%s",(

                                                                                                                                                                        self.var_dep.get(),
                                                                                                                                                                        self.var_course.get(),
                                                                                                                                                                        self.var_year.get(),
                                                                                                                                                                        self.var_semester.get(),
                                                                                                                                                                        self.var_std_name.get(),
                                                                                                                                                                        self.var_div.get(),
                                                                                                                                                                        self.var_roll.get(),
                                                                                                                                                                        self.var_gender.get(),
                                                                                                                                                                        self.var_dob.get(),
                                                                                                                                                                        self.var_email.get(),
                                                                                                                                                                        self.var_phone.get(),
                                                                                                                                                                        self.var_address.get(),
                                                                                                                                                                        self.var_teacher.get(),
                                                                                                                                                                        self.var_radio1.get(),
                                                                                                                                                                        self.va_std_id.get()
    
                                                                                                                                                                        ))
                else:
                    if not Upadate:
                        return
                messagebox.showinfo("Success","Student details successfully update completed",parent=self.root)
                conn.commit()
                self.fetch_data()
                conn.close()       
            except Exception as es:
                messagebox.showerror("Error",f"Due To:{str(es)}",parent=self.root)


                #Delete Funcation#
    def delete_data(self):
        if self.va_std_id.get()=="":
            messagebox.showerror("Error","Student Id must be required",parent=self.root)
        else:
            try:
                delete=messagebox.askyesno("Student Delete Page","Do You want to delete thise student",parent=self.root)
                if delete>0:
                    conn=mysql.connector.connect(host="localhost",username="root",password="Indianrrp@123",database="codewithrohit")
                    my_cursor=conn.cursor()
                    sql="delete from Student where Student_id=%s"
                    val=(self.va_std_id.get(),)
                    my_cursor.execute(sql,val)
                else:
                    if not delete:
                        return
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Delete","Successfully deleted student details",parent=self.root)   
            except Exception as es:
                messagebox.showerror("Error",f"Due To:{str(es)}",parent=self.root)




        
    
if __name__ == "__main__":
    root=Tk()
    obj=Student(root)
    root.mainloop()