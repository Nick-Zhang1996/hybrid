# test script, convert figure to gif
import os
import matplotlib.pyplot as plt
from PIL import Image

def resolveLogname():
    # setup log file
    # log file will record state of the vehicle for later analysis
    logFolder = "./gifs/"
    logPrefix = "iteration"
    logSuffix = ".gif"
    no = 1
    while os.path.isfile(logFolder+logPrefix+str(no)+logSuffix):
        no += 1

    log_no = no
    logFilename = logFolder+logPrefix+str(no)+logSuffix
    return logFilename

frame_vec = []
for i in range(10):
    fig, ax = plt.subplots()
    ax.plot(i,0,'o')
    ax.axis([0,10,-1,1])
    fig.canvas.draw()
    frame = Image.frombytes('RGB',
    fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    frame_vec.append(frame)
    plt.show()

gif_filename = resolveLogname()
frame_vec[0].save(fp=gif_filename,format='GIF',append_images=frame_vec,save_all=True,duration = 200,loop=0)
