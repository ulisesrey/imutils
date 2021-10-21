import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


start, end, num = 0, 200, 10000
t = np.linspace(start, end, num)
print(t[137])
#load pc1
pc1=np.sin(t)

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

line, = plt.plot(t,pc1, lw=2)
ax.set_xlabel('Time [s]')

# Make a horizontal slider to control the frequency.
axcolor = 'lightgoldenrodyellow'
start_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
start_slider = Slider(
    ax=start_ax,
    label='Starting time',
    valmin=0,
    valmax=1900,
    valinit=500,
    valstep=1
)

end_ax = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
end_slider = Slider(
    ax=end_ax,
    label='ending time',
    valmin=2000,
    valmax=3000,
    valinit=2500,
    valstep=1
)


# The function to be called anytime a slider's value changes
def update(val):
    #line.set_ydata(np.sin(np.linspace(int(start_slider.val), int(end_slider.val), num)))
    #line.set_ydata(np.sin(np.linspace(int(start_slider.val), int(end_slider.val),num)))

    # print(int(start_slider.val))
    # print(int(end_slider.val))
    #print(t[int(end_slider.val)])

    line.set_data(np.linspace(0,100,100),np.sin(np.linspace(int(start_slider.val),int(end_slider.val),100)))
    # start_c=int(start_slider.val)
    # end_c=int(end_slider.val)
    # print(start_c)
    # print(end_c)
    # line.set_ydata(t[start_c:end_c])
    #fig.canvas.draw_idle()

# register the update function with each slider
start_slider.on_changed(update)
end_slider.on_changed(update)

plt.show()