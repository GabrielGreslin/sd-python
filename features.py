__author__ = 'Gabriel'

'''
This is a template feature
'''


def templateFeat(x):  # x is an array with the 64 raw input
    assert (len(x) == 64), "Wrong input size"

    return [];  # a list of new features

def create_tab(t):
    s = 8
    r = []
    for i in range(0,8):
        r.append(t[i*s:(i+1)*s])
    return r

def sidePoints(x):  # x is an array with the 64 raw input
    assert (len(x) == 64), "Wrong input size"

    e = create_tab(x)

    o = []
    top = 0
    bottom = 0
    right = 0
    left = 0
    a = 0 # 00
    b = 0 # 01
    c = 0 # 11
    d = 0 # 10

    ai = 0
    aj = 0
    an = 0

    bi = 0
    bj = 0
    bn = 0

    ci = 0
    cj = 0
    cn = 0

    di = 0
    dj = 0
    dn = 0

    for i in range(0,8):
        o.extend(e[i])
        for j in range(0,8):
            r = e[i][j]
            if i < 4:
                top += r
                if j < 4:
                    a += r
                    ai += r * i
                    aj += r * j
                    an += r
                else:
                    bi += r * i
                    bj += r * j
                    bn += r
                    b += r
            else:
                bottom += r
                if j < 4:
                    di += r * i
                    dj += r * j
                    dn += r
                    d += r
                else:
                    ci += r * i
                    cj += r * j
                    cn += r
                    c += r
            if j < 4:
                left += r
            else:
                right += r
    v = []
    #av = [top,bottom,left,right,a,b,c,d]
    #av = [a,b,c,d]
    av = []
    if an > 0:
        av += [ai/an,aj/an]
    else:
        av += [0,0]
    if bn > 0:
        av += [bi/bn,bj/bn]
    else:
        av += [0,0]
    if cn > 0:
        av += [ci/cn,cj/cn]
    else:
        av += [0,0]
    if dn > 0:
        av += [di/dn,dj/dn]
    else:
        av += [0,0]

    av += [top,bottom,left,right,a,b,c,d]
    av += [a,b,c,d]

    v += av

    return v
