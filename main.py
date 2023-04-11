import math
import numpy as np
import sympy
from sympy import *
from tinydb import TinyDB, Query
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
from matplotlib import cm

def runlength(w):
    k = 1
    rl = []
    for i in range(len(w))[1:]:
        if w[i]-w[i-1] != 0:
            rl.append(k)
            k = 1
        else:
            k += 1
    rl.append(k)
    return rl


def compare_sequence(a, ap, b, bp):
    al = len(a)
    bl = len(b)
    for i in range(math.lcm(al, bl)):
        ia = i if i < al or ap == 0 else (al - ap) + (i- (al - ap)) % ap
        ib = i if i < bl or bp == 0 else (bl - bp) + (i- (bl - bp)) % bp
        va = math.inf
        vb = math.inf
        if ia < al:
            va = a[ia]
        if ib < bl:
            vb = b[ib]
        if va < vb:
            return pow(-1, i+1)
        elif va > vb:
            return pow(-1, i)
    return 0


def right_conjugate(s):
    t = s[:]
    if t[-1] == 1:
        t[-2] += 1
        t.pop()
    else:
        t[-1] -= 1
        t.append(1)
    return t


def center_point_ric(f1, f2, alpha):
    p = runlength(f1)
    s = runlength(f2)

    if compare_sequence([0] + p, len(p), alpha, 0) != -1:
        return p
    if compare_sequence(alpha, 0, [0] + right_conjugate(s) + s[::-1], len(s)) != -1:
        return s
    c = f1 + f2
    c_seq = runlength(c)
    r = compare_sequence(alpha, 0, c_seq, 0)
    if r == 0:
        return c_seq
    elif r == 1:
        return center_point_ric(c, f2, alpha)
    else:
        return center_point_ric(f1, c, alpha)


def rational_to_seq(a, b):
    s = []
    while b != 1:
        q, mod = divmod(a, b)
        a = b
        b = mod
        s.append(q)
    s.append(a)
    return s


def center_point(p, q):
    s = rational_to_seq(p, q)
    n = 2
    while True:
        r = compare_sequence(s, 0, [0, n], 0)
        if r == 0:
            return s
        elif r == 1:
            break
        n += 1
    return center_point_ric([0] * (n - 1) + [1], [0] * (n - 2) + [1], s)


def c_alpha(alpha, x):
    return floor(-x**(-1)+1-alpha)


def phi(alpha, p):
    c = c_alpha(alpha, p[1])
    x = simplify((c - p[0])**(-1))
    y = Integer(0)
    if p[1] != 0:
        y = simplify(-p[1]**(-1)-c)
    return x, y


def orbit(alpha, p, n):
    old = p
    r = [(float(p[0].evalf(5)), float(p[1].evalf(5)))]
    for i in range(n):
        p = phi(alpha, p)
        if p == old:
            break
        r.append((float(p[0].evalf(5)), float(p[1].evalf(5))))
        old = p

    return r


def fill_intermediate_points(l):
    ll = len(l)-1
    r = [0] * (ll * 2)
    for i in range(ll):
        r[i*2] = l[i]
        r[i*2+1] = (l[i+1][0], l[i][1])
    return r


def attractor(alpha):
    alpha = Rational(alpha)

    c = center_point(alpha.p, alpha.q)
    y = -continued_fraction_reduce([0, c])
    kl = list(reversed(c))
    x = continued_fraction_reduce([0, kl])
    n, m = sum(c[::2]), sum(c[1::2])
    xs = orbit(alpha, (x, alpha), m)
    ys = orbit(alpha, (y, alpha - 1), n)

    ys = sorted(ys)
    ys.append((float(x.evalf(5)), 0))
    ys = fill_intermediate_points(ys)

    xs = sorted(xs, reverse=True)
    xs.append((float(y.evalf(5)), 0))
    xs = fill_intermediate_points(xs)

    return xs, ys


def g(a, b, c, d):
    return ln(b*d+1)-ln(a*d+1)-ln(b*c+1)+ln(a*c+1)


def entropy(alp):
    xs, ys = attractor(alp)

    sol = 0
    for i in range(2, len(xs), 2):
        sol += g(xs[i+1][0], xs[i][0], alp-1, xs[i][1])

    for i in range(2, len(ys), 2):
        sol += g(ys[i][0], ys[i+1][0], ys[i][1], alp)

    sol += g(xs[1][0], ys[1][0], alp-1, alp)

    return float((pi**2/(3*sol)).evalf(5))


def induced_density_k(alpha,s):
    xs, ys = attractor(alpha)
    x0 = [(*x,0) for x in xs[::2]]
    x1 = [(*x,1) for x in xs[1::2]]
    y0 = [(*y,10) for y in ys[::2]]
    y1 = [(*y,11) for y in ys[1::2]]
    t = x0 + x1 + y0 + y1

    t=sorted(t,key=lambda k: (k[1],k[0]))

    r = []
    ca = ys[0][0]
    cb = ys[1][0]
    for i in range(2,len(t)):
        if t[i][2]==10:
            b=t[i][0]
            y=t[i][1]
            if len(r)==0 or r[-1][0] != y:
                r.append([y,ca,b])
        elif t[i][2]==11:
            cb = t[i][0]
        elif t[i][2]==0:
            ca = t[i][0]
        elif t[i][2]==1:
            a=t[i][0]
            y=t[i][1]
            r.append([y,a,cb])

    al = math.floor(alpha * 1000) / 1000
    m = gi(alpha-1,al-1.1,s)
    x = np.arange(al - 1.1, al + 0.1, s)
    y = np.zeros(len(x))

    for j in r:
        c = gi(j[0],al-1.1,s)
        xc = x[m:c]
        a = j[1]
        b = j[2]
        y[m:c] = list((b-a)/((a*xc+1)*(b*xc+1)))
        m = c

    plt.plot(x, y)
    plt.savefig("density_k.png", dpi=300)
    plt.close()
    #plt.ylim(bottom=0, top=3)
    #plt.savefig("induced_density_k.png",dpi=300)

    """
    y = k + np.sqrt(r ** 2 - (x - h) ** 2)
    d = (b-a)/((a*x+1)*(b*x+1))
    """

Todo = Query()
db = TinyDB('db.json')


def fill_data_db():
    config = db.get(Todo.type == "config")
    step = config['step']
    x = config['last'] - step
    while x > step:
        db.insert({'alpha': x, 'entr': entropy(x)})
        db.update({'last': x}, Todo.type == "config")
        x -= step


def plot_entropy():
    entropy = db.search(Todo.alpha.exists())
    print(entropy)
    xs, ys = [], []
    for v in entropy:
        xs.append(v['alpha'])
        ys.append(v['entr'])

    plt.plot(xs, ys)
    plt.ylabel('Entropy')
    plt.xlabel('Alpha')
    plt.show()


def get_xy(l):
    return [i[0] for i in l], [i[1] for i in l]


def interactive_plot():
    plt.subplots_adjust(bottom=0.35)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    l, = plt.plot([0], [0])

    axalpha = plt.axes([0.25, 0.15, 0.65, 0.03])

    alpha = Slider(axalpha, 'alpha', 0.0, 0.5, 4/15, valstep=0.01)

    def update_interactive_plot(_):
        a = alpha.val
        t = attractor(a)
        j = t[0]+t[1]+[t[0][0]]
        l.set_data(*get_xy(j))

    alpha.on_changed(update_interactive_plot)

    plt.show()


def show_region(alpha=4/15):
    xs, ys = attractor(alpha)

    plt.figure(figsize=(4, 4))
    for i in range(2, len(ys), 2):
        plt.plot(*get_xy([ys[i], ys[i + 1], (ys[i + 1][0], alpha), (ys[i][0], alpha), ys[i]]))

    for i in range(2, len(xs), 2):
        plt.plot(*get_xy([xs[i], xs[i + 1], (xs[i + 1][0], alpha-1), (xs[i][0], alpha-1), xs[i]]))

    plt.plot(*get_xy([xs[1],(xs[1][0], alpha-1),ys[1],(ys[1][0],alpha),xs[1]]))

    plt.savefig("attractor_regions.png", dpi=300)
    plt.close()


def check_ergodicity(alpha=4/15):
    alpha = Rational(alpha)
    px = []
    py = []

    for i in range(1000):
        x = Rational(random.random() * 0.6)
        y = Rational(-random.random() * 0.6)
        pxt,pyt = get_xy(orbit(alpha, (x, y), 100))
        px += pxt
        py += pyt

    plt.scatter(px, py, 0.3)
    plt.plot()#("ergodicity.png", dpi=400)


def plot(alpha=4/15):
    t = attractor(alpha)
    j = t[0]+t[1]+[t[0][0]]

    plt.figure(figsize=(4, 4))
    plt.plot(*get_xy(j))
    plt.savefig("attractor.png", dpi=300)
    plt.close()


def qumterval(s):
    r = runlength(s)
    sx = [0] + right_conjugate(r) + [r[::-1]]
    dx = [0, r]
    return (continued_fraction_reduce(sx),continued_fraction_reduce(dx))


def qumtervals(n):
    qumtervals = []
    fn = [[0], [1]]
    for i in range(n):
        l = len(fn)
        for x in range(l - 1):
            fn.insert(2 * x + 1, fn[2 * x] + fn[2 * x + 1])

    for i in fn[1:-1]:
        qumtervals.append(qumterval(i))
    return qumtervals


def semicircle(r, h, k):
    x0 = h - r
    x1 = h + r
    n = math.ceil(abs(x0-x1)/0.00001)
    if n<4:
        return None
    x = np.linspace(x0,x1,n,True)
    y = k + np.sqrt(r**2 - (x - h)**2)
    return x, y


def plot_qumtervals():
    qs = qumtervals(7)

    xs = []
    ys = []
    for q in qs:
        r = float(abs((q[0]-q[1])/2).evalf(5))
        c = float(abs((q[0]+q[1])/2).evalf(5))
        d = semicircle(r, c, 0)
        if d is not None:
            xs += list(d[0])
            ys += list(d[1])

    plt.plot(xs,ys,linewidth=0.5)
    plt.axis('scaled')
    plt.savefig('qumterval.png',dpi=500)


def k_alpha(alpha,x):
    if x == 0:
        return Integer(0)
    c = c_alpha(alpha, x)
    return (-x ** (-1) - c)

# get_index
def gi(x, start, step):
    return round((x-start)/step)


def num_density_k(alpha,s,rep):
    a = math.floor(alpha * 1000) / 1000
    x = np.arange(a - 1.1, a+0.1, s)
    st = np.zeros(len(x))

    alp = Rational(alpha)

    for i in range(rep):
        p = k_alpha(alp, Rational(alpha - 1 + random.random()))
        while p != 0:
            j = gi(p, a - 1.1, s)
            st[j] += 1
            p = k_alpha(alp, p)

    stf = np.copy(st)
    for j in range(2,8):
        for u in range(1,j):
            if math.gcd(j,u)==1:
                v = u/j
                if v > a-1 and v < a:
                    k = gi(v,a-1.1,s)
                    med = (stf[k - 1] + stf[k + 1]) / 2
                    stf[k]=med
                if -v > a-1 and -v < a:
                    k = gi(-v, a - 1.1, s)
                    med = (stf[k - 1] + stf[k + 1]) / 2
                    stf[k]=med

    sm = sum(st) * s / 1.07
    st = list(map(lambda x: x / sm, st))

    smf = sum(stf) * s / 1.07
    stf = list(map(lambda x: x / smf, stf))

    plt.plot(x, st)
    plt.savefig("num_density_k.png", dpi=300)
    plt.close()

    plt.plot(x, stf)
    plt.savefig("num_density_k_filtered.png", dpi=300)
    plt.close()


def num_density_phi(alpha, s, rep):
    a = math.floor(alpha * 1000) / 1000
    alpha = Rational(alpha)
    xs, ys = attractor(alpha)
    l, r = float(ys[0][0]), float(xs[0][0])

    x = np.arange(l-0.1, r+0.1, s)
    y = np.arange(a-1.1, a+0.1, s)
    X, Y = np.meshgrid(x, y)
    Z = X * 0

    for i in range(rep):
        p = phi(alpha, (Rational(random.uniform(l,r)), Rational(alpha - 1 + random.random())))
        while p[1] != 0:
            Z[gi(p[1], alpha - 1.1, s),gi(p[0], l - 0.1, s)] += 1
            p = phi(alpha, p)

    Zf = np.copy(Z)
    for j in range(2,8):
        for u in range(1,j):
            if math.gcd(j,u)==1:
                v = u/j
                if v > a-1 and v < a:
                    k = gi(v,a-1.1,s)
                    for d in range(len(Z[k])):
                        med = (Zf[k - 1,d] + Zf[k + 1,d]) / 2
                        Zf[k,d]=med
                if -v > a-1 and -v < a:
                    k = gi(-v, a - 1.1, s)
                    for d in range(len(Z[k])):
                        med = (Zf[k - 1, d] + Zf[k + 1, d]) / 2
                        Zf[k, d] = med

    plt.figure(figsize=(4,4))
    plt.pcolormesh(X, Y, Zf)
    plt.savefig("num_density_phi_filtered2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.pcolormesh(X, Y, Z)
    plt.savefig("num_density_phi2.png", dpi=300)
    plt.close()

    r = np.zeros(len(y))
    for i in range(len(y)):
        r[i] = np.sum(Z[i])

    sm = sum(r) * s / 1.07
    r = list(map(lambda x: x / sm, r))

    rf = np.zeros(len(y))
    for i in range(len(y)):
        rf[i] = np.sum(Zf[i])

    smf = sum(rf) * s / 1.07
    rf = list(map(lambda x: x / smf, rf))

    plt.plot(y,r)
    plt.savefig("num_induced_density_k2.png",dpi=300)
    plt.close()

    plt.plot(y,rf)
    plt.savefig("num_induced_density_k_filtered2.png",dpi=300)
    plt.close()


def density_phi(alpha, step, margin):
    alpha = Rational(alpha)
    xs, ys = attractor(alpha)

    l, r = float(ys[0][0]), float(xs[0][0])
    ml, mr = l - margin, r + margin
    mb, mt = float(alpha) - 1 - margin, float(alpha) + margin
    x = np.arange(ml, mr, step)
    y = np.arange(mb, mt, step)
    X, Y = np.meshgrid(x, y)
    Z = X * 0

    for i in range(2, len(ys), 2):
        il = gi(ys[i][0], ml, step)
        ir = gi(ys[i + 1][0], ml, step)
        ib = gi(ys[i][1], mb, step)
        it = gi(alpha, mb, step)
        Z[ib:it,il:ir] = 1/(1 + X[ib:it,il:ir]*Y[ib:it,il:ir])**2

    for i in range(2, len(xs), 2):
        il = gi(xs[i+1][0], ml, step)
        ir = gi(xs[i][0], ml, step)
        ib = gi(alpha-1, mb, step)
        it = gi(xs[i][1], mb, step)
        Z[ib:it, il:ir] = 1 / (1 + X[ib:it, il:ir] * Y[ib:it, il:ir]) ** 2

    il = gi(xs[1][0], ml, step)
    ir = gi(ys[1][0], ml, step)
    ib = gi(alpha - 1, mb, step)
    it = gi(alpha, mb, step)
    Z[ib:it, il:ir] = 1 / (1 + X[ib:it, il:ir] * Y[ib:it, il:ir]) ** 2

    f, ax = plt.subplots(figsize=(4, 4))
    plt.pcolormesh(X,Y,Z)
    plt.savefig("density_phi.png",dpi=300)



def main():

    alpha = 4/15
    s = 0.005
    rep = 500
    margin = 0.1
    #num_density_phi(alpha, s, rep)
    #density_phi(alpha,s,margin)
    #plot()
    #show_region()
    #num_density_k(alpha, s, rep)
    #induced_density_k(alpha,s)
    #num_density_phi(alpha, s, rep)
    interactive_plot()


main()