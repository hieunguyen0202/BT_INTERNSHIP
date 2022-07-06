from flask import Flask, render_template, redirect, url_for, request, session, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import timedelta
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, DateTime, Boolean, Enum
from datetime import datetime
from enum import Enum as UserEnum
from flask_admin import Admin, AdminIndexView,expose, BaseView
from flask_admin.contrib.sqla import ModelView
from app.admin import *
from os import path
import hashlib
import query
# day la train model
import cv2
from flask_mail import Mail, Message
from random import randrange







app = Flask(__name__)
app.config["SECRET_KEY"]="LDVNLSNVL"
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:hieu26082001@localhost/bt_internship?charset=utf8mb4"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.permanent_session_lifetime = timedelta(minutes=30)

db = SQLAlchemy(app=app)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME']='cubin26082001@gmail.com'
app.config['MAIL_PASSWORD']='sbifkgoeiqhkquhn'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)


# class MyAdmin(AdminIndexView):
#     @expose('/')
#     def index(self):
#         return self.render('admin/home.html')
class Homepage(BaseView):
    @expose('/')
    def index(self):
        return self.render('layout/Homepage.html')

admin = Admin(app=app, name='Manage User', template_mode='bootstrap4')
Predict = 0


class UserRole(UserEnum):
    ADMIN = 1
    USER = 2

class User(db.Model):
    user_id = db.Column(db.Integer, primary_key= True)
    username = Column(db.String(50), nullable=False)
    password = Column(db.String(50), nullable=False)
    date_birth = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    active = Column(Boolean, default=True)
    department = Column(db.String(100))
    status = Column(db.String(100))
    joined_date = Column(DateTime, default=datetime.now())
    user_role = Column(db.String(100))
    def __init__(self, username,password, email,date_birth, department,user_role):
        self.username = username
        self.password = password
        self.email = email
        self.date_birth = date_birth
        self.department =department
        self.user_role = user_role





admin.add_view(ModelView(User, db.session, name='Manage'))
admin.add_view(Homepage(name="Home Page"))
def check_login(email, password):
    if email and password:
        password = str(hashlib.md5(password.strip().encode('utf-8')).hexdigest())

        return User.query.filter(User.email.__eq__(email.strip()),
                                 User.password.__eq__(password)).first()

    return None

def check_user_admin(email,user_role):
    if email and user_role:
        return User.query.filter(User.email.__eq__(email.strip()),User.user_role.__eq__(user_role.strip())).first()

    return None

def check_resetmail(email):
    if email:
        return User.query.filter(User.email.__eq__(email.strip())).first()

    return None


@app.route('/forget_pass', methods=['GET', 'POST'])
def forget():
    ktra_us =""
    PIN = randrange(000000, 999999)
    session["PIN"] = str(PIN)
    if request.method == 'POST':
        mail_smtp = request.form["mail_smtp"]
        ktra_us = mail_smtp + str(PIN)
        session["ktra_us"] = ktra_us
        msg = Message("PIN TO RESET YOUR PASSWORD", sender='noreply@demo.com',
                      recipients=[mail_smtp])
        msg.body = "Please enter 6-digit PIN below to change your password: " + str(PIN)
        mail.send(msg)
        return redirect(url_for("reset"))

    return render_template("layout/forget_password.html")

@app.route('/reset_pass', methods=['GET', 'POST'])
def reset():
    if request.method == 'POST':
        email_rs = request.form["email"]
        new_pass = request.form["newpass"]
        PIN_NEW = request.form["PIN"]
        reset_user = User.query.filter_by(email=email_rs).first()
        PIN_SS = session["ktra_us"]
        ktra_us_2 = email_rs + PIN_NEW
        if reset_user and  ktra_us_2==PIN_SS:
            new_pass_haslib = str(hashlib.md5(new_pass.strip().encode('utf-8')).hexdigest())
            reset_pass = User.query.filter_by(password=new_pass_haslib).first()
            if reset_pass:
                return render_template("layout/login.html")
            else:
                reset_user.password = new_pass_haslib
                db.session.commit()
                return render_template("layout/login.html")
        else:
            return render_template("layout/resetpass.html")



    return render_template("layout/resetpass.html")

@app.route('/home')
def home():
    return render_template("layout/Homepage.html",display_signup = True)


@app.route('/manage')
def manage():
    return render_template("layout/manage.html")

@app.route("/signup", methods = ['get', 'post'])
def sign_up():
    if request.method == "POST":
        user_name = request.form["name"]
        password = request.form["password"]
        email = request.form["email"]
        datebirth = request.form["date_birth"]
        department = request.form["department"]
        user_role = request.form["user_role"]

        password = hashlib.md5(str(password).encode("utf-8")).hexdigest()
        session.permanent = True
        if email:
            session["user"] = user_name
            session["password"] = password
            session["email"] = email

            found_user = User.query.filter_by(email=email).first()
            if found_user:
                session["email"] = found_user.email
            else:
                user = User(user_name,password, email,datebirth,department,user_role)
                db.session.add(user)
                db.session.commit()
                flash("You sign up successfully", "info")
                return render_template("layout/login.html", user =email, VALID=True)

    return render_template("layout/signup.html", display_signup=False)


@app.route('/login', methods=["POST", "GET"])
def login_hello():

    if request.method == "POST":

        email = request.form.get('email')
        password = request.form.get('password')
        user_role = request.form.get('user_role')
        session["user1"] = email

        user = check_login(email=email, password=password)
        ktra_admin = check_user_admin(email=email,user_role=user_role)

        if user:
            if ktra_admin:
                session["valid"] = user_role
                return redirect(url_for("user", user1=email, valid=user_role))
                flash("Ban da dang nhap thanh cong với Admin", "info")
            else:
                session["valid"] = "User"
                return redirect(url_for("user", user1=email, valid="USER"))
                flash("Ban da dang nhap thanh cong", "info")
        else:
            flash("Username or Password invalid", "info")
            return render_template("layout/login.html", VALID=False)


    return render_template("layout/login.html",VALID=True,display_signup = True)


@app.route('/logout')
def hello_logout():
    flash("Ban da logout thanh cong", "info")
    session.pop("user", None)
    return redirect(url_for("login_hello"))


from keras.models import load_model
import numpy as np
import tensorflow as tf


model = load_model("model_train_face.h5")
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
img = cv2.imread('static/images/146442277_756401631941367_7119192181113375909_n.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



@app.route("/user", methods = ['get', 'post'])
def user():
    camera = False
    name_user = ""
    dep_user = ""
    # labl = user.query.get(1)
    # labl = labl.status
    flash("ban da dang nhap thanh cong")
    if "valid" in session:
        role = session["valid"]
    if "user1" in session:
        name = session["user1"]
        face = face_detector.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)
        if request.method == "POST":
            camera_m = request.form.get('camera')
            if camera_m == "oncam":
                camera = True
            else:
                camera = False

        for (x, y, w, h) in face:
            face_img = img[y: y + h, x: x + w]
            cv2.imwrite('Hienthi.jpg', face_img)
            test_image = tf.keras.utils.load_img('Hienthi.jpg', target_size=(150, 150, 3))
            test_image = tf.keras.utils.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            predict_image = model.predict(test_image)[0][0]
            # Predict = Process_face()
            if predict_image == 1:
                name_list = User.query.get(2)
                name_user = name_list.username
                dep_user = name_list.department
            else:
                name_list = User.query.get(1)
                name_user = name_list.username
                dep_user = name_list.department

        return render_template("layout/user.html", user1= name, display=True,msg="ok",display_signup=True,name_user = name_user,dep_user = dep_user,came=camera,valid=role)






#
#
# @app.route("/user", methods = ['get', 'post'])
# def user():
#     email = None
#     if "user" in session:
#         name = session["user"]
#         if request.method == "POST":
#             if not request.form["email"] and request.form["name"]:
#                 User.query.filter_by(name = name).delete()
#                 # db.session.commit()
#                 flash("ban da xoa user")
#                 return redirect(url_for("hello_logout"))
#             else:
#                 email = request.form["email"]
#                 session["email"] = email
#                 found_user = User.query.filter_by(name = name).first()
#                 found_user.email = email
#                 # db.session.commit()
#                 flash("email da duoc sua doi")
#         elif "email" in session:
#             email = session["email"]
#         return render_template("layout/user.html", user= name, email = email)
#     else:
#         flash("Ban chua login", "info")
#         # return render_template("demo2.html", massage="XIN CHAO %s !!" % name)
#         return redirect(url_for("login_hello"))
#
#




if __name__ == "__main__":

    app.run(debug=True)