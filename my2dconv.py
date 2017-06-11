
import numpy as np

def myconv2d(img, kernel, strides):
    km,kn = kernel.shape
    x,y = img.shape

    if(x != y):
        print 'Reshape the image to same # of cols and rows'

    if(km == kn):
        # Flip the kernel along both the axis
        kernel = np.flip(kernel, 0)
        kernel = np.flip(kernel, 1)

        # Zero padding to ensure size of output is same as input
        padding = int((km - strides) / 2)

        # Pad the input image
        inp_image_padding = np.pad(img, padding, mode='constant', constant_values=0)

        new_img = np.zeros((x,y))
        for i in range(x):
            for j in range(y):
                new_img[i][j] = np.sum(inp_image_padding[i:i+km, j:j+kn]*kernel)
    return new_img


img = np.array([[1,0.5,1,0.5,1],
       [0.5,1,0.5,1,0.5],
       [1,0.5,1,0.5,1],
       [0.5,1,0.5,1,0.5],
       [1,0.5,1,0.5,1]], np.float32)

k = np.array([[1,2,3],
          [4,5,6],
          [7,8,9]], np.float32)

strides = 1

new_img = myconv2d(img, k, strides)
