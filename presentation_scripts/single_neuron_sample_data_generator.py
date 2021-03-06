import random
import matplotlib.pyplot as plt
import numpy


def dcdw_dcdb(xs, ys, w, b):
    total_w = 0
    total_b = 0
    for x, y in zip(xs, ys):
        total_w += 2 * x * ((b + (w * x)) - y)
        total_b += 2 * ((b + (w * x)) - y)
    return total_w, total_b

# TODO: look into https://stackoverflow.com/questions/7130474/3d-vector-field-in-matplotlib

def main():
    m = -0.7
    c = 3
    xs = []
    ys = []
    for _ in range(10):
        x = random.randrange(100)
        x = x/10. - 5.
        y = m * x + c
        y = numpy.random.normal(y, 0.35)
        xs.append(x)
        ys.append(y)
    
    # xs = [-2.4, 4.300000000000001, -2.0, -3.7, -0.7999999999999998, 3.5, 4.0, -4.6, 0.9000000000000004, -0.20000000000000018, 1.5999999999999996, 0.0, 1.7000000000000002, 1.0999999999999996, -3.4, -4.9, -4.6, 1.0, -0.40000000000000036, -2.7]
    # ys = [4.759293605346915, 0.13560844662018068, 4.409004322428809, 5.951626163611069, 3.535332546158398, 0.4345618659832912, 0.19630767796951268, 6.180286480925455, 2.3955829525915457, 3.2350001397287773, 1.982982490017593, 3.160544994893823, 2.1304015064786053, 2.2043924820289433, 5.3960684168484265, 6.283015091058907, 6.204641342258453, 2.342950919346456, 3.3687053847893824, 4.926563213159792]
    # xs = [2.5999999999999996, 1.0999999999999996, -1.2999999999999998, 3.5, 1.9000000000000004, 1.7000000000000002, 1.9000000000000004, 2.5, 0.5, 2.0999999999999996]
    # ys = [1.1652351304587412, 2.705812861324486, 3.6677296482192934, 0.5936572691701745, 1.6920558774240564, 1.811585400937681, 2.153338300485198, 0.8427302922065774, 2.5413553300250222, 1.977319644626561]
    
    xs = [-2.0, 3.6999999999999993, 4.699999999999999, 1.0, 3.4000000000000004, 1.9000000000000004, 0.5, 3.6999999999999993, -3.7, -1.4]
    ys = [4.416856407182695, 0.47057967159614744, -0.29887819819469424, 1.4013832483099857, 0.4316035832132115, 1.4101057523285265, 2.0118609471559328, 0.5242922714864773, 5.620627159141899, 3.494602189431038]
    
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    
    ms, cs = numpy.polyfit(xs, ys, 1)

    plt.xlim((-5, 5))
    plt.ylim((ms * 5 + cs, ms * -5 + cs))

    plt.savefig("presentation/img/single-neuron-example-data.svg")


    regression_line = ax.plot([-5, 5], [ms * -5 + cs, ms * 5 + cs], color='red', label='y = {:.2f}x + {:.2f}'.format(ms, cs))
    ax.legend(handles=regression_line)
    
    plt.savefig("presentation/img/single-neuron-example-data-with-regression-line.svg")

    w = 1
    b = 1
    mu = 0.1


    weights = [(w, b)]
    print("({}, {})".format(w, b))
    for i in range(100):
        dw, db = dcdw_dcdb(xs, ys, w, b)
        w = w - mu * dw
        b = b - mu * db
        weights.append((w, b))


    print(xs)
    print(ys)
    for x, y in zip(xs, ys):
        print("<tr><th>{:.1f}</th><th>{:.1f}</th></tr>".format(x, y))


    def print_weight(index):
        w, b = weights[index]
        print("<li class=\"fragment\" value=\"{}\">$w$ = {:.4g}, $b$ = {:.4g}</li>".format(index + 1, w, b))

    # mu is 0.01/0.1
    print_weight(0)
    print_weight(1)
    print_weight(2)
    print_weight(3)
    print_weight(4)
    print_weight(7)
    print_weight(17)
    print_weight(25)
    print_weight(35)

    # mu is 0.001
    # print_weight(0)
    # print_weight(1)
    # print_weight(2)
    # print_weight(4)
    # print_weight(7)
    # print_weight(17)
    # print_weight(56)
    # print_weight(125)
    # print_weight(356)
    # print_weight(357)

if __name__ == "__main__":
    main()