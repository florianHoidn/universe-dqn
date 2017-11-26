import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def plot_rewards(plot_file = "csv_for_plotting.txt"):
    """ This code is based on Sentdex's tutorial (see https://pythonprogramming.net/python-matplotlib-live-updating-graphs/)
    and modified slightly so that various graphs can be plotted dynamically from a suitable file.
    """

    fig = plt.figure()
    fig.suptitle("rewards vs. approximated rewards")
    ax = fig.add_subplot(1,1,1)

    def animate(i):
        pull_data = open(plot_file,"r").read()
        data_array = pull_data.split('\n')    
        headers = data_array[0].split(',')
        value_arrays = [[] for i in range(len(headers))]
        for each_line in data_array[1:-1]:
            values_to_plot = each_line.split(',')
            for i in range(len(values_to_plot)):
                value_arrays[i].append(float(values_to_plot[i]))
        ax.clear()
        for i in range(len(value_arrays)):
            ax.plot(list(range(len(value_arrays[i]))), value_arrays[i], label=headers[i])
        plt.legend()    
        
    ani = animation.FuncAnimation(fig, animate, interval=500) # Update plot every 0.5 seconds.
    plt.show()

if __name__ == "__main__":
    plot_rewards()