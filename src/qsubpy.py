import subprocess

names = ('A', 'B', 'C')

par = ()

for n in names:
    p = subprocess.Popen(['qsub', '-N', n, 'job.py'])
    p.wait()