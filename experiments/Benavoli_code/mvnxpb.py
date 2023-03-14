from numpy import *
from scipy.special import erfc

eps = 2.2204e-16
"""
This is a Python reimplementation of Gentz' algortihm
http://www.math.wsu.edu/faculty/genz/software/matlab/mvnxpb.m
"""


def Phi(z):
    return erfc(-z / sqrt(2)) / 2


def phi(z):
    return exp(-(z**2) / 2) / sqrt(dot(2, pi))


def bvnmmg(a=None, b=None, sg=None):
    cx = sqrt(sg[0, 0])
    cy = sqrt(sg[1, 1])
    r = sg[1, 0] / (dot(cx, cy))
    xl = a[0] / cx
    xu = b[0] / cx
    yl = a[1] / cy
    yu = b[1] / cy
    Ex, Ey, p = bvnmom(xl, xu, yl, yu, r)
    Ex = dot(Ex, cx)
    Ey = dot(Ey, cy)
    # end bvnmmg
    return Ex, Ey, p


# @function
def bvnmom(xl, xu, yl, yu, r):
    # varargin = bvnmom.varargin
    # nargin = bvnmom.nargin

    rs = 1 / sqrt(1 - r**2)

    #  A function for computing bivariate normal probability moments;
    #  bvnmom calculates expected values Ex, Ey, for bivariate normal x, y,
    #   with xl < x < xu and yl < y < yu, and correlation coefficient r.

    #  This function uses generalizations of formulas found in
    #    Moments of a Truncated Bivariate Normal Distribution
    #    S. Rosenbaum, JRSS B 23 (1961) pp. 405-408.

    # print(xl,xu,yl,yu,r)
    p = bvnu(xl, yl, r) - bvnu(xu, yl, r) - bvnu(xl, yu, r) + bvnu(xu, yu, r)
    if xl == -inf and yl == -inf and xu == inf and yu == inf:
        Ex = 0
        Ey = 0
    else:
        if xl == -inf and xu == inf and yl == -inf:
            Ey = -phi(yu)
            Ex = 0
        else:
            if xl == -inf and xu == inf and yu == inf:
                Ey = phi(yl)
                Ex = 0
            else:
                if xl == -inf and yl == -inf and yu == inf:
                    Ex = -phi(xu)
                    Ey = 0
                else:
                    if xu == inf and yl == -inf and yu == inf:
                        Ex = phi(xl)
                        Ey = 0
                    else:
                        if xl == -inf and xu == inf:
                            Ey = phi(yl) - phi(yu)
                            Ex = 0
                        else:
                            if yl == -inf and yu == inf:
                                Ex = phi(xl) - phi(xu)
                                Ey = 0
                            else:
                                if xl == -inf and yl == -inf:
                                    Phiyxu = Phi(dot((yu - dot(r, xu)), rs))
                                    pPhixy = dot(-phi(xu), Phiyxu)
                                    Phixyu = Phi(dot((xu - dot(r, yu)), rs))
                                    pPhiyx = dot(-phi(yu), Phixyu)
                                else:
                                    if xu == inf and yu == inf:
                                        Phiyxl = Phi(dot(-(yl - dot(r, xl)), rs))
                                        pPhixy = dot(phi(xl), Phiyxl)
                                        Phixyl = Phi(dot(-(xl - dot(r, yl)), rs))
                                        pPhiyx = dot(phi(yl), Phixyl)
                                    else:
                                        if xl == -inf and yu == inf:
                                            Phiyxu = Phi(dot(-(yl - dot(r, xu)), rs))
                                            pPhixy = dot(-phi(xu), Phiyxu)
                                            Phixyl = Phi(dot((xu - dot(r, yl)), rs))
                                            pPhiyx = dot(phi(yl), Phixyl)
                                        else:
                                            if xu == inf and yl == -inf:
                                                Phiyxl = Phi(dot((yu - dot(r, xl)), rs))
                                                pPhixy = dot(phi(xl), Phiyxl)
                                                Phixyu = Phi(
                                                    dot(-(xl - dot(r, yu)), rs)
                                                )
                                                pPhiyx = dot(-phi(yu), Phixyu)
                                            else:
                                                if xl == -inf:
                                                    Phiyxu = Phi(
                                                        dot((yu - dot(r, xu)), rs)
                                                    ) - Phi(dot((yl - dot(r, xu)), rs))
                                                    pPhixy = dot(-phi(xu), Phiyxu)
                                                    Phixyl = Phi(
                                                        dot((xu - dot(r, yl)), rs)
                                                    )
                                                    Phixyu = Phi(
                                                        dot((xu - dot(r, yu)), rs)
                                                    )
                                                    pPhiyx = dot(phi(yl), Phixyl) - dot(
                                                        phi(yu), Phixyu
                                                    )
                                                else:
                                                    if xu == inf:
                                                        Phiyxl = Phi(
                                                            dot((yu - dot(r, xl)), rs)
                                                        ) - Phi(
                                                            dot((yl - dot(r, xl)), rs)
                                                        )
                                                        pPhixy = dot(phi(xl), Phiyxl)
                                                        Phixyl = Phi(
                                                            dot(-(xl - dot(r, yl)), rs)
                                                        )
                                                        Phixyu = Phi(
                                                            dot(-(xl - dot(r, yu)), rs)
                                                        )
                                                        pPhiyx = dot(
                                                            phi(yl), Phixyl
                                                        ) - dot(phi(yu), Phixyu)
                                                    else:
                                                        if yl == -inf:
                                                            Phiyxl = Phi(
                                                                dot(
                                                                    (yu - dot(r, xl)),
                                                                    rs,
                                                                )
                                                            )
                                                            Phiyxu = Phi(
                                                                dot(
                                                                    (yu - dot(r, xu)),
                                                                    rs,
                                                                )
                                                            )
                                                            pPhixy = dot(
                                                                phi(xl), Phiyxl
                                                            ) - dot(phi(xu), Phiyxu)
                                                            Phixyu = Phi(
                                                                dot(
                                                                    (xu - dot(r, yu)),
                                                                    rs,
                                                                )
                                                            ) - Phi(
                                                                dot(
                                                                    (xl - dot(r, yu)),
                                                                    rs,
                                                                )
                                                            )
                                                            pPhiyx = dot(
                                                                -phi(yu), Phixyu
                                                            )
                                                        else:
                                                            if yu == inf:
                                                                Phiyxl = Phi(
                                                                    dot(
                                                                        -(
                                                                            yl
                                                                            - dot(r, xl)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                )
                                                                Phiyxu = Phi(
                                                                    dot(
                                                                        -(
                                                                            yl
                                                                            - dot(r, xu)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                )
                                                                pPhixy = dot(
                                                                    phi(xl), Phiyxl
                                                                ) - dot(phi(xu), Phiyxu)
                                                                Phixyl = Phi(
                                                                    dot(
                                                                        (
                                                                            xu
                                                                            - dot(r, yl)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                ) - Phi(
                                                                    dot(
                                                                        (
                                                                            xl
                                                                            - dot(r, yl)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                )
                                                                pPhiyx = dot(
                                                                    phi(yl), Phixyl
                                                                )
                                                            else:
                                                                Phiyxl = Phi(
                                                                    dot(
                                                                        (
                                                                            yu
                                                                            - dot(r, xl)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                ) - Phi(
                                                                    dot(
                                                                        (
                                                                            yl
                                                                            - dot(r, xl)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                )
                                                                Phiyxu = Phi(
                                                                    dot(
                                                                        (
                                                                            yu
                                                                            - dot(r, xu)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                ) - Phi(
                                                                    dot(
                                                                        (
                                                                            yl
                                                                            - dot(r, xu)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                )

                                                                pPhixy = dot(
                                                                    phi(xl), Phiyxl
                                                                ) - dot(phi(xu), Phiyxu)
                                                                Phixyl = Phi(
                                                                    dot(
                                                                        (
                                                                            xu
                                                                            - dot(r, yl)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                ) - Phi(
                                                                    dot(
                                                                        (
                                                                            xl
                                                                            - dot(r, yl)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                )

                                                                Phixyu = Phi(
                                                                    dot(
                                                                        (
                                                                            xu
                                                                            - dot(r, yu)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                ) - Phi(
                                                                    dot(
                                                                        (
                                                                            xl
                                                                            - dot(r, yu)
                                                                        ),
                                                                        rs,
                                                                    )
                                                                )
                                                                pPhiyx = dot(
                                                                    phi(yl), Phixyl
                                                                ) - dot(phi(yu), Phixyu)

                                Ex = pPhixy + dot(r, pPhiyx)
                                Ey = dot(r, pPhixy) + pPhiyx

    Ex = Ex / p
    Ey = Ey / p
    return Ex, Ey, p
    #  end bvnmom


# @function
def bvnu(dh, dk, r):
    # print(dh,dk,r)
    # varargin = bvnu.varargin
    # nargin = bvnu.nargin

    # BVNU
    #  A function for computing bivariate normal probabilities.
    #  bvnu calculates the probability that x > dh and y > dk.
    #    parameters
    #      dh 1st lower integration limit
    #      dk 2nd lower integration limit
    #      r   correlation coefficient
    #  Example: p = bvnu( -3, -1, .35 )
    #  Note: to compute the probability that x < dh and y < dk,
    #        use bvnu( -dh, -dk, r ).

    #   Author
    #       Alan Genz
    #       Department of Mathematics
    #       Washington State University
    #       Pullman, Wa 99164-3113
    #       Email : alangenz@wsu.edu

    #    This function is based on the method described by
    #        Drezner, Z and G.O. Wesolowsky, (1989),
    #        On the computation of the bivariate normal inegral,
    #        Journal of Statist. Comput. Simul. 35, pp. 101-107,
    #    with major modifications for double precision, for |r| close to 1,
    #    and for Matlab by Alan Genz. Minor bug modifications 7/98, 2/10.

    if dh == inf or dk == inf:
        p = 0
    else:
        if dh == -inf:
            if dk == -inf:
                p = 1
            else:
                p = Phi(-dk)
        else:
            if dk == -inf:
                p = Phi(-dh)
            else:
                if abs(r) < 0.3:
                    ng = 1
                    lg = 3

                    w = zeros((lg, ng))
                    x = zeros((lg, ng))
                    w[:, 0] = array([0.171324492379, 0.360761573048, 0.467913934573]).T
                    x[:, 0] = array([0.932469514203, 0.661209386466, 0.238619186083]).T
                else:
                    if abs(r) < 0.75:
                        ng = 2

                        lg = 6
                        w = zeros((lg, ng))
                        x = zeros((lg, ng))
                        w[0:3, 1] = array(
                            [0.0471753363865, 0.106939325995, 0.160078328543]
                        ).T
                        w[3:6, 1] = array(
                            [0.203167426723, 0.233492536538, 0.249147045813]
                        ).T
                        x[0:3, 1] = array(
                            [0.981560634247, 0.90411725637, 0.769902674194]
                        ).T
                        x[3:6, 1] = array(
                            [0.587317954287, 0.367831498998, 0.125233408511]
                        ).T
                    else:
                        ng = 3
                        lg = 10
                        w = zeros((lg, ng))
                        x = zeros((lg, ng))
                        w[0:3, 2] = array(
                            [0.0176140071392, 0.0406014298004, 0.0626720483341]
                        ).T
                        w[3:6, 2] = array(
                            [0.0832767415767, 0.101930119817, 0.118194531962]
                        ).T
                        w[6:9, 2] = array(
                            [0.131688638449, 0.142096109318, 0.149172986473]
                        ).T
                        w[9, 2] = 0.152753387131
                        x[0:3, 2] = array(
                            [0.993128599185, 0.963971927278, 0.912234428251]
                        ).T
                        x[3:6, 2] = array(
                            [0.839116971822, 0.74633190646, 0.636053680727]
                        ).T
                        x[6:9, 2] = array(
                            [0.510867001951, 0.373706088715, 0.227785851142]
                        ).T
                        x[9, 2] = 0.0765265211335
                h = copy(dh)
                k = copy(dk)
                hk = dot(h, k)
                bvn = 0
                if abs(r) < 0.925:
                    hs = (dot(h, h) + dot(k, k)) / 2
                    asr = arcsin(r)
                    for i in arange(0, lg).reshape(-1):
                        # print(x)
                        sn = sin(dot(asr, (1 - x[i, ng - 1])) / 2)
                        bvn = bvn + dot(
                            w[i, ng - 1], exp((dot(sn, hk) - hs) / (1 - dot(sn, sn)))
                        )
                        sn = sin(dot(asr, (1 + x[i, ng - 1])) / 2)
                        bvn = bvn + dot(
                            w[i, ng - 1], exp((dot(sn, hk) - hs) / (1 - dot(sn, sn)))
                        )
                    bvn = dot(bvn, asr) / (dot(4, pi))
                    bvn = bvn + dot(Phi(-h), Phi(-k))
                else:
                    twopi = dot(2, pi)
                    if r < 0:
                        k = -k
                        hk = -hk
                    if abs(r) < 1:
                        as_ = dot((1 - r), (1 + r))
                        a = sqrt(as_)
                        bs = (h - k) ** 2
                        c = (4 - hk) / 8
                        d = (12 - hk) / 16
                        asr = -(bs / as_ + hk) / 2
                        if asr > -100:
                            bvn = dot(
                                dot(a, exp(asr)),
                                (
                                    1
                                    - dot(dot(c, (bs - as_)), (1 - dot(d, bs) / 5)) / 3
                                    + dot(dot(dot(c, d), as_), as_) / 5
                                ),
                            )
                        if hk > -100:
                            b = sqrt(bs)
                            sp = dot(sqrt(twopi), Phi(-b / a))
                            bvn = bvn - dot(
                                dot(dot(exp(-hk / 2), sp), b),
                                (1 - dot(dot(c, bs), (1 - dot(d, bs) / 5)) / 3),
                            )
                        a = a / 2
                        for i in arange(0, lg).reshape(-1):
                            for is_ in arange(-1, 1, 2).reshape(-1):
                                xs = (a + dot(dot(a, is_), x[i, ng - 1])) ** 2
                                rs = sqrt(1 - xs)
                                asr = -(bs / xs + hk) / 2
                                if asr > -100:
                                    sp = 1 + dot(dot(c, xs), (1 + dot(d, xs)))
                                    ep = (
                                        exp(dot(-hk, xs) / (dot(2, (1 + rs) ** 2))) / rs
                                    )
                                    bvn = bvn + dot(
                                        dot(dot(a, w[i, ng - 1]), exp(asr)), (ep - sp)
                                    )
                        bvn = -bvn / twopi
                    if r > 0:
                        bvn = bvn + Phi(-maximum(h, k))
                    else:
                        if h >= k:
                            bvn = -bvn
                        else:
                            if h < 0:
                                L = Phi(k) - Phi(h)
                            else:
                                L = Phi(-h) - Phi(-k)
                            bvn = L - bvn
                p = maximum(0, minimum(1, bvn))

    return p
    #   end bvnu


def mvnxpb(r, a, b):
    n = r.shape[0]
    c = copy(r)
    ap = copy(a)
    bp = copy(b)
    ep = 1e-10

    d = sqrt(maximum(diag(c), 0))
    for i in range(n):
        if d[i] > 0:
            ap[i] = ap[i] / d[i]
            bp[i] = bp[i] / d[i]
            c[:, i] = c[:, i] / d[i]
            c[i, :] = c[i, :] / d[i]

    # Dynamically determine Cholesky factor,
    #    permuting variables to minimize outer integrals
    y = zeros((n, 1))
    p = 1
    pb = 1
    D = eye(n)
    for k in range(0, n):
        im = k
        ckk = 0
        dem = 1
        s = 0
        for i in range(k, n):
            if c[i, i] > eps:
                cii = sqrt(maximum(c[i, i], 0))
                if i > 0:
                    s = atleast_2d(c[i, :k]).dot(y[:k])
                ai = (ap[i] - s) / cii
                bi = (bp[i] - s) / cii
                # print(bi)
                de = Phi(bi) - Phi(ai)
                if de <= dem:
                    ckk = cii
                    dem = de
                    am = ai
                    bm = bi
                    im = i
        if im > k:
            km = arange(0, k)
            ip = arange(im + 1, n)
            ki = arange(k + 1, im - 1 + 1)
            # km = arange(0,k-1); ip = arange(im+1,n); ki = arange(k+1,im-1);
            ap[[im, k]] = ap[[k, im]]
            bp[[im, k]] = bp[[k, im]]
            ctmp = copy(c)
            c[im, km] = ctmp[k, km]
            c[k, km] = ctmp[im, km]
            if len(ip) > 0:
                ctmp = copy(c)
                # c[ip][[im,k]]=c[ip][[k,im]]
                c[ip, im] = ctmp[ip, k]
                c[ip, k] = ctmp[ip, im]
            t = c[ki, k]
            c[ki, k] = c[im, ki].T
            c[im, ki] = t.T
            c[im, im] = c[k, k]
        if ckk > dot(ep, k):
            c[k, k] = ckk
            c[k, k + 1 : n] = 0
            for i in range(k + 1, n):
                c[i, k] = c[i, k] / ckk
                c[i, k + 1 : i + 1] = c[i, k + 1 : i + 1] - c[i, k] * (
                    c[k + 1 : i + 1, k].T
                )
            if abs(dem) > ep:
                y[k] = (phi(am) - phi(bm)) / dem
            else:
                y[k] = (am + bm) / 2
                if am < -9:
                    y[k] = bm
                else:
                    if bm > 9:
                        y[k] = am

        else:
            # print(c)

            c[k:n, k] = 0
            y[k] = (ap[k] + bp[k]) / 2
        p = dot(p, dem)
        # print(c)

        if mod(k + 1, 2) == 0:
            u = c[k - 1, k - 1]
            v = c[k, k]
            w = c[k, k - 1]
            c[k - 1 :, k - 1 : k + 1] = c[k - 1 :, k - 1 : k + 1].dot(
                array([[1 / u, 0], [-w / (u * v), 1 / v]])
            )
            ab = ap[k - 1 : k + 1]
            bb = bp[k - 1 : k + 1]
            cb = 0
            cb = 0
            if k > 2:
                cb = c[k - 1 : k + 1, 0 : k - 2 + 1].dot(y[: k - 2 + 1])
            sg = array([[u**2, dot(u, w)], [dot(u, w), w**2 + v**2]])
            D[k - 1 : k + 1, k - 1 : k + 1] = sg
            Ex, Ey, bv = bvnmmg(ab - cb, bb - cb, sg)
            # print(atleast_1d(Ex),atleast_1d(Ey),bv)
            pb = dot(pb, bv)
            y[k - 1 : k + 1] = array([[atleast_1d(Ex)[0]], [atleast_1d(Ey)[0]]])

    if mod(n, 2) == 1:
        pb = dot(pb, dem)
    return pb
    # end mvnxpb
