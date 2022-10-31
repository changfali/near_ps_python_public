import numpy as np
import mathutils
import math
from scipy.io import savemat

# a = np.random.rand(100,2)
# print(np.max(a), np.min(a))
# b = (a>0.5).astype(np.float32)
# print(b)
# b = np.zeros((100,3,3))
# b[:,0,2] = -np.ones(100)

# a = np.ones((2,500,3))
# b = 2*np.ones((500,3))
# print((np.sum(a*b, axis=-1)).shape)
# idx = np.where(b>0)
# print(idx[0])
# print(np.where(np.random.rand(100)>0.2)[0].shape)
# a = np.random.randn(1000,1)
# print(np.concatenate((a,a,a,a), axis=1).shape)

# a = np.random.randn(500,3)
# b = np.random.randn(16,3)
# print((a[None,:,:]*b[:,None,:]).shape)

def get_light_coord():
    start_pos = [   mathutils.Vector(( 77.85/1000, -300.36/1000, 844.29/1000)), 
                    mathutils.Vector((180.05/1000, -547.05/1000, 590.47/1000)), 
                    mathutils.Vector((157.31/1000, -267.45/1000, 844.29/1000)), 
                    mathutils.Vector((259.51/1000, -514.14/1000, 590.47/1000))  ]

    light_coord = []
    mat_rot = mathutils.Matrix.Rotation(math.radians(45.0), 4, 'Z')
    for pos in start_pos:
        for _ in range(8):
            light_coord.append(pos)
            pos = mat_rot@pos
    return light_coord
# savemat('light_p.mat', {'lights_p':np.array(get_light_coord())*1000})
# print(np.array(get_light_coord())*1000)
# print(4**3.2)
# a = np.random.randn(100,3,3)
# b = np.random.randn(32,100,3,1)
# print((a@b).shape)
# a = np.ones((3,3)); a[:,1] = 2; a[:,2] = 3
# print(a, '\n\n', np.tile(a, [2,2]))
# print(np.arange(1000))
# from scipy import sparse as sp
# a = sp.csr_matrix((np.ones(1000), (np.arange(1000), np.tile(np.arange(100), 10))))
# print(a.shape)

# a = np.ones(1000)
# # b = a.copy()
# # a /= 5
# # print(b)
# a.reshape(20,50)[:5,:5] = 100
# print(a)
import cv2 
# a = np.random.randn(153,35)
# print(cv2.resize(a, [4,19]).shape)
# img = cv2.imread('testData/out/n_2.png')
# h,w,c = img.shape
# cv2.imwrite('test1.png', img.reshape(w,h,c))

# a = np.meshgrid(np.arange(100), np.arange(200))
# print(a[0].shape)
# i = cv2.imread('testData/spoon/imgs/DSCF6870_spoon.png', -1)
# print(np.max(i), np.min(i))
imgs = np.load('testData/yellow/mid_3h.npy').transpose(3,0,1,2)[:,::16,::16,:]
print(np.max(imgs))
for i in range(imgs.shape[0]):
    cv2.imwrite('out/changfa/temp/{}.png'.format(i),(imgs[i,:]*65535).astype(np.uint16))


