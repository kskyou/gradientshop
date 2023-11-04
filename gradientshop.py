DESCRIPTION = '''This is a partial implementation of GradiantShop
https://grail.cs.washington.edu/projects/gradientshop/demos/gs_paper_TOG_2009.pdf
by Kevin You. This implementation is intended to be a lightweight alternative in Python.

Image are assumed to be non-linear sRGB. The following filters are implemented:
1. Non-photorealistic rendering (default): Blurs out detail while preserving edges to achieve artistic effect.
2. Saliency sharpening: Sharpens edges.
3. Sparse interpolation: Takes in guide image and user image with user scribbles applied to guide image.
   Creates a new image with colors from user scribbles and gradients from luminance of guide image.

Constants: Constants include c1, b, c2, sigma, and can be modified in the __main__ body.
In particular, the sigma constant is worth playing around to control the strength of the NPR filter.

Deviations: The most important difference between this implementation and GradientShop is that this
implementation uses a Sobel edge detector instead of one based on second order Gaussian steerable filters.
Another difference is that length estimation via message passing is optional. Good images have been
achieved with none or a lower number of passes. Additionally, sigma is calculated depending on the number of passes.
Finally, for sparse interpolation the target gradients are those of luminance of the guide image.'''

import argparse
import math
import numpy as np
import scipy
import skimage as ski
import datetime

DEBUG = False
MESSAGE = 0

def convolve(ker, image):
    return scipy.signal.convolve2d(image, ker, mode='same', boundary='symm')

def conjugate_gradient(A, b, x_init, epsilon, N):

    x_star = x_init
    r = b - A(x_star) 
    d = r
    delta_new = np.sum(np.multiply(r, r))
    n = 0
    while math.sqrt(delta_new) > epsilon and n < N:
        q = A(d)
        eta = delta_new / np.sum(np.multiply(d, q))
        x_star = x_star + eta * d
        r = r - eta * q
        delta_old = delta_new
        delta_new = np.sum(np.multiply(r, r))
        d = r + delta_new / delta_old * d
        n += 1

    if (n < N):
        print("----Used " + str(n) + " iterations")
    else:
        print("----Warning: Used all " + str(n) + " iterations. Residue " + str(math.sqrt(delta_new)) + " remains")

    return x_star

def construct_system(d, g, w, epsilon, N): 

    # Matrix operation of A^TA where we are minimizing ||Af-b|| 
    def ATA(f):

        data_f = np.multiply(w[0], np.multiply(w[0], f))

        shift_left = np.pad(f,((0,0),(0,1)), mode='constant')[:, 1:]
        f_x_int = np.multiply(w[1], shift_left - f)
        
        shift_right = np.pad(np.multiply(w[1],f_x_int),((0,0),(1,0)), mode='constant')[:, :-1]
        f_x = shift_right - np.multiply(w[1], f_x_int)

        shift_up = np.pad(f,((0,1),(0,0)), mode='constant')[1:, :]
        f_y_int = np.multiply(w[2], shift_up - f)
        
        shift_down = np.pad(np.multiply(w[2],f_y_int),((1,0),(0,0)), mode='constant')[:-1, :]
        f_y = shift_down - np.multiply(w[2], f_y_int)

        return data_f + f_x + f_y 

    # Constructing A^Tb where we are minimizing ||Af-b||
    g_x = np.multiply(w[1], g[0])
    shift_right = np.pad(np.multiply(w[1],g_x),((0,0),(1,0)), mode='constant')[:, :-1]
    b_x = shift_right - np.multiply(w[1], g_x)

    g_y = np.multiply(w[2], g[1])
    shift_down = np.pad(np.multiply(w[2],g_y),((1,0),(0,0)), mode='constant')[:-1, :]
    b_y = shift_down - np.multiply(w[2], g_y)

    ATb = np.multiply(w[0], np.multiply(w[0], d)) + b_x + b_y

    # Solve A^TAx = A^Tb via conjugate gradient method
    return conjugate_gradient(ATA, ATb, d, epsilon, N)


class Filter():

    def d(self):
        return self.d

    def g(self):
        return self.g

    def w(self):
        return self.w

    def ux(self, u):
        shift_left = np.pad(u,((0,0),(0,1)), mode='constant')[:, 1:]
        return shift_left - u
        
    def uy(self, u):
        shift_up = np.pad(u,((0,1),(0,0)), mode='constant')[1:, :]
        return shift_up - u


    def edge(self, u, R):
        
        # Local edge
        C2 = convolve(np.array([[1,0,-1],[2,0,-2],[1,0,-1]]), u)
        C3 = convolve(np.array([[1,2,1],[0,0,0],[-1,-2,-1]]), u)

        arg = lambda c : math.atan2(c[0], c[1]) 
        p_0 = np.apply_along_axis(arg, 0, np.stack([C2, C3]))
        p_m = np.sqrt(C2 ** 2 + C3 ** 2)

        # Edge normalization
        kernel = np.ones((2*R+1,2*R+1))
        p_sum = scipy.signal.convolve2d(p_m, kernel, mode="same", boundary='symm')
        p_sq_sum = scipy.signal.convolve2d(p_m ** 2, kernel, mode="same", boundary='symm')
        p_num = scipy.signal.convolve2d(np.ones(p_m.shape), kernel, mode="same", boundary='symm')

        mu_w = np.divide(p_sum, p_num)
        sigma_w = np.sqrt(np.divide(p_sq_sum - np.divide(p_sum ** 2, p_num), p_num) + 0.00001)
        p_hat = np.divide(p_m - mu_w, sigma_w + 0.00001)

        if DEBUG:
            ski.io.imsave("c2.png", np.clip(255*np.abs(C2)/np.max(np.abs(C2)), 0, 255).astype(np.uint8))
            ski.io.imsave("c3.png", np.clip(255*np.abs(C3)/np.max(np.abs(C3)), 0, 255).astype(np.uint8))
            ski.io.imsave("phat.png", np.clip(255*np.abs(p_hat)/np.max(np.abs(p_hat)), 0, 255).astype(np.uint8))

        # Message passing
        if MESSAGE > 0:
            print("--Begin message passing")
        m0 = np.zeros(u.shape)
        m1 = np.zeros(u.shape)

        def interp(m, mnew, x, y, xq, yq):
            if xq < 0 or yq < 0 or xq >= m.shape[1]-1 or yq >= m.shape[0]-1:
                return
            xqf = math.floor(xq)
            yqf = math.floor(yq)
            L = [(xqf, yqf), (xqf+1, yqf), (xqf, yqf+1), (xqf+1, yqf+1)]
            for q in L:
                qx = q[0]
                qy = q[1]
                weight_alpha = (1-abs(xq-qx)) * (1-abs(yq-qy))
                diff_theta = abs(p_0[y,x] - p_0[qy,qx])
                weight_theta = math.exp(- 1/2 * (diff_theta/(0.63)) ** 2)
                mnew[y,x] += weight_alpha * weight_theta * (m[qy, qx] + p_hat[qy, qx])

        for i in range(MESSAGE):
            print("----Message iteration " + str(i))
            m0new = np.zeros(u.shape)
            m1new = np.zeros(u.shape)
            for y in range(u.shape[0]):
                for x in range(u.shape[1]):
                    angle = p_0[y,x]
                    x0 = x + math.sqrt(2)*math.cos(angle)
                    x1 = x - math.sqrt(2)*math.cos(angle)
                    y0 = y + math.sqrt(2)*math.sin(angle)
                    y1 = y - math.sqrt(2)*math.sin(angle)
                    interp(m0, m0new, x, y, x0, y0)
                    interp(m1, m1new, x, y, x1, y1)
            m0 = m0new
            m1 = m1new
        elp = p_hat + m0 + m1
        return [elp * np.average(np.abs(p_hat)) / np.average(np.abs(elp)), p_0]


class Sharpen(Filter):
    
    def __init__(self, u, c1, b, c2):
        ux = self.ux(u)
        uy = self.uy(u)
        [elp, eo] = self.edge(u, 3)
        cos_sq = (np.vectorize(math.cos)(eo)) ** 2
        sin_sq = (np.vectorize(math.sin)(eo)) ** 2
        sx = np.multiply(ux, np.multiply(sin_sq, elp))
        sy = np.multiply(uy, np.multiply(cos_sq, elp))
        gx = ux + c2 * sx
        gy = ux + c2 * sy

        if DEBUG:
            s = np.sqrt(sx ** 2 + sy ** 2)
            ski.io.imsave("saliency.png", np.clip(255*s/np.max(s), 0, 255).astype(np.uint8))

        w = lambda p : math.pow(abs(p[0] - p[1]) + 1, -b)
        w1 = np.apply_along_axis(w, 0, np.stack([ux, gx]))
        w2 = np.apply_along_axis(w, 0, np.stack([uy, gy]))

        self.d = u
        self.g = np.stack([gx, gy])
        self.w = np.stack([c1 * np.ones((u.shape[0], u.shape[1])), w1, w2])


class NPR(Filter):

    def __init__(self, u, c1, b, c2, sigma):
        ux = self.ux(u)
        uy = self.uy(u)
        [elp, eo] = self.edge(u, 3)

        n_weight = lambda p : c2 * (1 - math.exp(-1/2 * (p/sigma) ** 2))
        n = np.vectorize(n_weight)(elp)
        cos_sq = (np.vectorize(math.cos)(eo)) ** 2
        sin_sq = (np.vectorize(math.sin)(eo)) ** 2
        gx = np.multiply(ux, np.multiply(sin_sq, n))
        gy = np.multiply(uy, np.multiply(cos_sq, n))

        if DEBUG:
            g = np.sqrt(gx ** 2 + gy ** 2)
            ski.io.imsave("gradient.png", np.clip(255*s/np.max(s), 0, 255).astype(np.uint8))

        w = lambda p : math.pow(abs(p[0] - p[1]) + 1, -b)
        w1 = np.apply_along_axis(w, 0, np.stack([ux, gx]))
        w2 = np.apply_along_axis(w, 0, np.stack([uy, gy]))

        self.d = u
        self.g = np.stack([gx, gy])
        self.w = np.stack([c1 * np.ones((u.shape[0], u.shape[1])), w1, w2])


class Interp(Filter):

    def __init__(self, u, lumi, mask, b, c2):
        ux = self.ux(lumi)
        uy = self.uy(lumi)
        [elp, eo] = self.edge(lumi, 3)

        cos_sq = (np.vectorize(math.cos)(eo)) ** 2
        sin_sq = (np.vectorize(math.sin)(eo)) ** 2
        sx = np.multiply(ux, np.multiply(sin_sq, elp))
        sy = np.multiply(uy, np.multiply(cos_sq, elp))

        if DEBUG:
            s = np.sqrt(sx ** 2 + sy ** 2)
            ski.io.imsave("saliency.png", np.clip(255*s/np.max(s), 0, 255).astype(np.uint8))

        w = lambda p : math.pow(abs(p) + c2, -b)
        w1 = np.vectorize(w)(sx)
        w2 = np.vectorize(w)(sy)

        self.d = u
        #self.g = np.zeros((2, u.shape[0], u.shape[1]))
        self.g = np.stack([ux, uy])
        self.w = np.stack([100 * mask, w1, w2])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=DESCRIPTION,formatter_class=argparse.RawTextHelpFormatter) 
    parser.add_argument('image_path', metavar='<image_path>', type=str)
    parser.add_argument('-npr', help='non-photorealistic rendering filter (default)', action='store_true')
    parser.add_argument('-sharpen', help='sharpening filter', action='store_true')
    parser.add_argument('-interp', help='sparse data interpolation filter', metavar='<guide_image>', action='store')
    parser.add_argument('-message', help='number of message passing iterations, default = 0', type=int, action='store')
    parser.add_argument('-downscale', help='downscale image as integer factor, default = 1', type=int, action='store')
    parser.add_argument('-debug', help='output intermediate gradient images', action='store_true')
    args = parser.parse_args()

    DEBUG = args.debug
    MESSAGE = 0 if args.message == None else args.message
    SCALE = 1 if args.downscale == None else args.downscale

    I_color = (ski.io.imread(args.image_path).astype(float) * (1/255))[::SCALE,::SCALE,:3]
    #ski.io.imsave("original.png", (np.clip(I_color, 0, 1) * 255).astype(np.uint8))
    #quit()
    
    I_result = None
    if args.npr or args.sharpen or not args.interp:
        if args.sharpen:
            image_filter = 'sharpen'
            print("Applying saliency sharpening filter to " + args.image_path)
        else:
            image_filter = 'npr'
            print("Applying non-photorealistic rendering filter to " + args.image_path)

        I_list = []
        for c in range(3):
            print("Color channel " + str(c))
            I = I_color[:,:,c] # Non-linear image 

            F = None
            if image_filter == 'npr':
                F = NPR(I, 0.03, 5, 3, 1.8) # c1=0.03, b=5, c2=3, sigma=1.8
            elif image_filter == 'sharpen':
                F = Sharpen(I, 0.05, 5, 1.5) # c1=0.05, b=5, c2=1.5

            print("--Begin solving system") # Solves minimization
            res = construct_system(F.d, F.g, F.w, 0.000001, 2000) # epsilon, N

            I_list.append(res)
        I_result = np.stack(I_list, axis=-1)

    else:
        image_filter = 'interp'
        print("Applying sparse interpolation filter to " + args.image_path)

        G_color = (ski.io.imread(args.interp).astype(float) * (1/255))[::SCALE,::SCALE,:3]
        assert(I_color.shape == G_color.shape)

        diff = lambda c : 0 if c[0] and c[1] and c[2] else 1
        Mask = np.apply_along_axis(diff, -1, np.equal(I_color, G_color))
        if DEBUG:
            ski.io.imsave("mask.png", (255 * Mask).astype(np.uint8))

        gamma = lambda c : 12.92 * c if c <= 0.0031308 else 1.055 * math.pow(c, 1/2.4) - 0.055
        gammainv = lambda c : c / 12.92 if c <= 0.0404482 else math.pow((c + 0.055)/1.055, 2.4)
        sRGBtoXYZ = np.array([[0.4124564,0.3575761,0.1804375],[0.2126729,0.7151522,0.0721750],[0.0193339,0.1191920,0.9503041]])
        #XYZtosRGB = np.array([[3.2404542,-1.5371385,-0.4985314],[-0.9692660,1.8760108,0.0415560],[0.0556434,-0.2040259,1.0572252]])
        G_lumi = np.vectorize(gamma)(np.apply_along_axis(sRGBtoXYZ.dot, -1, np.vectorize(gammainv)(G_color))[:,:,1])

        I_list = []
        I_preblend = []
        for c in range(3):
            print("Color channel " + str(c))
            I = I_color[:,:,c] # Non-linear image 

            F = Interp(I, G_lumi, Mask, 0.05, 5) # c2 = 0.05, b=5

            print("--Begin solving system") # Solves minimization
            res = construct_system(F.d, F.g, F.w, 0.000001, 5000) # epsilon, N
            I_preblend.append(res)

            print("--Begin blending image")  # Poisson blend the minimized image and the guide image on the scribbled region
            nonzero = lambda z : 0 if z == 0 else 1
            Mask_wide = np.vectorize(nonzero)(convolve(np.ones((7, 7)),Mask))
            def masked_laplace(f):
                return np.multiply(Mask_wide, convolve(np.array([[0,1,0],[1,-4,1],[0,1,0]]), f))
            res_blend = conjugate_gradient(masked_laplace, masked_laplace(G_lumi), res, 0.00000001, 2000)

            I_list.append(res_blend)
        I_result = np.stack(I_list, axis=-1)
        if DEBUG:
            I_preblend_result = np.stack(I_preblend, axis=-1)
            ski.io.imsave("preblend.png", (np.clip(I_preblend_result, 0, 1) * 255).astype(np.uint8))

    [image_path, image_type] = args.image_path.rsplit('.', 1)
    image_name = image_path + "_" + image_filter + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "." + image_type
    ski.io.imsave(image_name, (np.clip(I_result, 0, 1) * 255).astype(np.uint8))
    print("Filtered image saved to " + image_name)
    
    if DEBUG:
        ski.io.imsave("difference.png", np.clip(255*np.abs(I_result-I_color), 0, 255).astype(np.uint8))


