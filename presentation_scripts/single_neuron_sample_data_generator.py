import random
import matplotlib.pyplot as plt
import numpy

def main():
    m = -0.7
    c = 3
    xs = []
    ys = []
    for _ in range(20):
        x = random.randrange(100)
        x = x/10. - 5.
        y = m * x + c
        y = numpy.random.normal(y, 0.1)
        xs.append(x)
        ys.append(y)
    
    
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    
    ms, cs = numpy.polyfit(xs, ys, 1)

    regression_line = ax.plot([-5, 5], [ms * -5 + cs, ms * 5 + cs], color='red', label='y = {:.1f}x + {:.1f}'.format(ms, cs))
    ax.legend(handles=regression_line)

    plt.savefig("presentation/img/single-neuron-example-data.svg")



if __name__ == "__main__":
    main()