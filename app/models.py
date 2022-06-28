from file_main import *
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Float, DateTime, Boolean, Enum
from datetime import datetime
from enum import Enum as UserEnum
from os import path

class UserRole(UserEnum):
    ADMIN = 1
    USER = 2

class User(db.Model):
    user_id = db.Column(db.Integer, primary_key= True)
    username = Column(db.String(50), nullable=False, unique=True)
    password = Column(db.String(50), nullable=False)
    email = db.Column(db.String(100))
    date_birth = db.Column(db.String(100))
    active = Column(Boolean, default=True)
    department = Column(db.String(100))
    status = Column(db.String(100))
    joined_date = Column(DateTime, default=datetime.now())
    user_role = Column(Enum(UserRole), default=UserRole.USER)
    def __init__(self, username,password, email, department):
        self.username = username
        self.password = password
        self.email = email
        self.department =department



if __name__ == "__main__":
    if not path.exists("user.db"):
        db.create_all(app=app)
        print("Created database")

