import hashlib

mk = hashlib.md5("123456789".encode("utf-8")).hexdigest()
print(mk)