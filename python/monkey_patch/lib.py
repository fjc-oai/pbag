import subprocess

def f():
    subprocess.call(['tail', '-f', '/var/log/daily.out'])