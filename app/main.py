from app import *
from file_main import User
p = User.query.get(1)

print(p.__dict__)