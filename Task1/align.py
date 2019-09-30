import numpy as np
import skimage.io as io

def alignn(im1, im2, limit):
    x_c=0
    y_c=0
    c_val = -np.inf
    [H,W] = np.shape(im2)
    x_coord = np.arange(0,W,1)
    y_coord = np.arange(0,H,1)
    for y in range(-limit,limit+1):
        for x in range(-limit,limit+1):
            x_coord1 = x_coord[(x_coord+x>=0) & (x_coord+x<W)]
            y_coord1 = y_coord[(y_coord+y>=0) & (y_coord+y<H)]
            im22 = im2[np.ix_(y_coord1,x_coord1)]
#             im22 = np.roll(np.roll(im2, -y, axis = 0), -x, axis = 1)
            if x<0:
                x1 = 0
                x2 = W+x
                if y<0:
#                     im11 = im1[0:H+y, 0:W+x]
                    y1=0
                    y2=H+y
                else:
#                     im11 = im1[y:, 0:W+x]
                    y1 = y
                    y2 = im1.shape[0]
            elif x>=0: 
                x1 = x
                x2 = im1.shape[1]
                if y<0:
#                     im11 = im1[0:H+y, x:]
                    y1=0
                    y2=H+y
                else:
#                     im11 = im1[y:, x:]
                    y1 = y
                    y2 = im1.shape[0]
            im11 = im1[y1:y2, x1:x2]        
# #             mse = np.mean((im11 - im22) ** 2)
# #             if m_val > mse:
# #                 m_val = mse
# #                 x_m = x
# #                 y_m = y 
           
            cor = corr(im11,im22)
            if c_val < cor:
                c_val = cor
                x_c = x
                y_c = y
#     print(y_m, x_m)
#     print('y_c_x_c',y_c, x_c)
    return [y_c,x_c]
    
def im_resize(im, c):
    imag = im
    while(c > 0):
        imag = imag[::2,::2]
        c-=1
    return imag
    
def my_range(start, end, step):
    while start >= end:
        yield start
        start += step
        
def corr(im11, im22):
    cor = np.mean((im11 - im11.mean()) * (im22 - im22.mean()))
#     stds = im11.std() * im22.std()
#     cor /= stds
    return np.max(cor)
    
def mse(im11, im22):
    return np.mean((im11 - im22) ** 2)
    
def align(im, g_coord):
    [Height,Width]=np.shape(im);
    Height = int(np.rint(Height/3));
    delta_x = int(np.rint(Width*0.05))
    delta_y = int(np.rint(Height*0.05))
    im1 = im[delta_y:(Height-delta_y),delta_x:(Width-delta_x)]
#     print(im1.shape)
    im2 = im[Height+delta_y:(2*Height-delta_y),delta_x:(Width-delta_x)]
#     print(im2.shape)
    im3 = im[2*Height +delta_y:(3*Height-delta_y),delta_x:(Width-delta_x)]
#     print(im3.shape)
    image = np.array([im1,im2,im3[0:im1.shape[0],:]])
    [imm1,imm2,imm3] = image
#     print(image.shape)
    
    [H,W] = image[0,:,:].shape
    k=1
    height = H;
    width = W;
    while (height>500 and width>500):
        height/=2
        width/=2
        k+=1
#     print(k)    
#     print('Processing...')    
    limit=16
    t_x=0
    t_y=0
    y_c=0
    x_c=0
    t_xx=0
    t_yy=0
    y_cc=0
    x_cc=0
    if k == 1:
        [t_y,t_x]=alignn(image[0,:,:], image[1,:,:], 15)
        [t_yy,t_xx]=alignn(image[2,:,:], image[1,:,:], 15)
    else:
        for i in my_range(k-1,2,-1):
            im1 = im_resize(imm1,i)
            im3 = im_resize(imm3,i)
            im2 = im_resize(imm2,i)
            
            [y_c,x_c] = alignn(im1, im2, limit)
            imm1 = np.roll(np.roll(imm1, (-y_c*(2**i)), axis = 0), (-x_c*(2**i)), axis = 1)
            t_x = t_x + x_c*(2**i)
            t_y = t_y + y_c*(2**i)
            
            [y_cc,x_cc] = alignn(im3, im2, limit)
            imm3 = np.roll(np.roll(imm3, (-y_cc*(2**i)), axis = 0), (-x_cc*(2**i)), axis = 1)
            t_xx = t_xx + x_cc*(2**i)
            t_yy = t_yy + y_cc*(2**i)
            limit = int(limit/2)

    [b_row, b_col] = [g_coord[0]-Height+t_y, g_coord[1]+t_x] 
    [r_row, r_col] = [g_coord[0]+Height+t_yy, g_coord[1]+t_xx]

    aligned_img = np.dstack((imm3, image[1,:,:], imm1))

    return aligned_img, (b_row, b_col), (r_row, r_col)