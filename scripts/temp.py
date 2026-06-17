from time import sleep
for i in range(100):
    print(f'{i}/100', end='\r', flush=True)
    sleep(0.1)
