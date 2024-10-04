'''
This is the script for running multiple seeds at a time.
If you want to test only one seed, you need to comment one line of code in each train.py and main.py. 
'''
import subprocess

for seed in range(10, 30):
    process1 = subprocess.Popen(['python', 'server.py'])

    process2 = subprocess.Popen(f'python train.py --seed={seed} --hospital=1', shell=True)
    process3 = subprocess.Popen(f'python train.py --seed={seed} --hospital=2', shell=True)

    process1.wait()
    process2.wait()
    process3.wait()

    process4 = subprocess.Popen(f'python main.py --seed={seed} --hospital=1', shell=True)
    process5 = subprocess.Popen(f'python main.py --seed={seed} --hospital=2', shell=True)

    process4.wait()
    process5.wait()
