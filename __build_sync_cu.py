import os
import shutil


try:
    os.makedirs('test_cu')
except:
    pass

for item in os.listdir('test'):
    if item.endswith('.cpp'):
        src = os.path.join('test', item)
        dst = os.path.join('test_cu', item.replace('.cpp', '.cu'))
        shutil.copyfile(src, dst)
