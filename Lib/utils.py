import os

def mkdirs(folder, permission):
    if not os.path.exists(folder):
        try:
            origin_umask = os.umask(0)
            os.makedirs(folder, permission, exist_ok=True)
        finally:
            os.umask(origin_umask)
