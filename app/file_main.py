from flask import Flask, render_template, redirect, url_for, request, session, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import timedelta
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, DateTime, Boolean, Enum
from datetime import datetime
from enum import Enum as UserEnum
from os import path
import hashlib
import query




app = Flask(__name__)
app.config["SECRET_KEY"]="LDVNLSNVL"
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:hieu26082001@localhost/csdl3?charset=utf8mb4"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.permanent_session_lifetime = timedelta(minutes=30)

db = SQLAlchemy(app=app)


class UserRole(UserEnum):
    ADMIN = 1
    USER = 2

class User(db.Model):
    user_id = db.Column(db.Integer, primary_key= True)
    username = Column(db.String(50), nullable=False, unique=True)
    password = Column(db.String(50), nullable=False)
    date_birth = db.Column(db.String(100))
    email = db.Column(db.String(100))
    active = Column(Boolean, default=True)
    department = Column(db.String(100))
    status = Column(db.String(100))
    joined_date = Column(DateTime, default=datetime.now())
    user_role = Column(Enum(UserRole), default=UserRole.USER)
    def __init__(self, username,password, email,date_birth, department):
        self.username = username
        self.password = password
        self.email = email
        self.date_birth = date_birth
        self.department =department




def check_login(username, password):
    if username and password:
        password = str(hashlib.md5(password.strip().encode('utf-8')).hexdigest())

        return User.query.filter(User.username.__eq__(username.strip()),
                                 User.password.__eq__(password)).first()

    return None


@app.route('/home')
def home():
    return render_template("layout/Homepage.html")


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
        password = hashlib.md5(str(password).encode("utf-8")).hexdigest()
        session.permanent = True
        if user_name:
            session["user"] = user_name
            session["password"] = password

            found_user = User.query.filter_by(username=user_name).first()
            if found_user:
                session["email"] = found_user.email
            else:
                user = User(user_name,password, email,datebirth,department)
                db.session.add(user)
                db.session.commit()
                flash("You sign up successfully", "info")
                return render_template("layout/login.html", user =user_name, VALID=True)

    return render_template("layout/signup.html")


@app.route('/login', methods=["POST", "GET"])
def login_hello():
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')
        session["user1"] = username
        user = check_login(username=username, password=password)
        if user:
            return redirect(url_for("user", user1=username))
            flash("Ban da dang nhap thanh cong", "info")
        else:
            flash("Username or Password invalid", "info")
            return render_template("layout/login.html", VALID=False)


    return render_template("layout/login.html",VALID=True)


@app.route('/logout')
def hello_logout():
    flash("Ban da logout thanh cong", "info")
    session.pop("user", None)
    return redirect(url_for("login_hello"))

@app.route("/user", methods = ['get', 'post'])
def user():
    # labl = user.query.get(1)
    # labl = labl.status
    flash("ban da dang nhap thanh cong")
    if "user1" in session:
        name = session["user1"]
        return render_template("layout/user.html", user1= name, display=True,msg="ok")


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