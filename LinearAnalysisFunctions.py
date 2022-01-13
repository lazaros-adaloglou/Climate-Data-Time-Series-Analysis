import matplotlib.pyplot as plt


# Plot Time Series.
def plot_timeseries(x, value='', title='', savepath='', zoomy=1):

    plt.plot(x, marker='x', linestyle='--', linewidth=1)
    plt.xlabel('Time (Days)')
    plt.ylabel(value)
    plt.ylim([-zoomy * abs(min(x)) - abs(min(x)), zoomy * abs(max(x)) + 0.2*abs(max(x))])
    if len(title) > 0:
        plt.title(title, x=0.5, y=1.0)
    if len(savepath) > 0:
        plt.savefig(f'{savepath}/{title} Time Series.png')


# Plot Histogram.
def plot_histogram(x, value, title='', savepath=''):

    plt.hist(x, alpha=0.8, rwidth=0.9)
    plt.xlabel(value)
    plt.title('Histogram')
    if len(title) > 0:
        plt.title(title, x=0.5, y=1.0)
    if len(savepath) > 0:
        plt.savefig(f'{savepath}/{title} Histogram.png')
