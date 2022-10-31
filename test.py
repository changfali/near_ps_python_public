import numpy as np
import cv2
import os
import json
import scipy.io as sio
import time

from semi_calib_opt import Semi_calib


############################################## test0727
# size = 16                                                                                                                    # 下采样比例
# z0 = 730
# scales = 2                                                                                                                    # 场景平均高度
# K_path = 'testData/spoon/camera.mat'                                                                                        # 相机内参
# lights_path = 'testData/0727/light_coord.npy'                                                                                    # 相机坐标系下的光源坐标
# imgs_path = 'testData/0727/png/'                                                                                            # 图片文件夹
# imgs_save_path = 'out/changfa/new_1/'                                                                                             # 结果保存路径
# save_current_result = True                                                                                                  # 是否保存中间结果
# cos4a=True                                                                                                                  # 原始图像是否进行cos4a的矫正

# semi_n = Semi_calib('data/config_opt.json')
# semi_n.scales=scales
# K = sio.loadmat(K_path)['K']
# # light_p = sio.loadmat(lights_path)['S']
# light_p = np.load(lights_path); light_p[:,1] = -light_p[:,1]; light_p[:,2] = z0 - light_p[:,2]
# light_d = light_p - np.array([[0,0,z0]]); light_d = light_d / np.linalg.norm(light_d, axis=1)[:,None]
# good_list = [0,1,2,3,4,5,6,7,10,11,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
# light_p = light_p[good_list,:]; light_d = light_d[good_list,:]
# imgs_dir = os.listdir(imgs_path); imgs_dir.sort(key = lambda x: x.split('_')[2])      # 注意图片名称
# imgs_dir_list = [imgs_path+x for x in imgs_dir]
# imgs = semi_n.load_imgs(imgs_dir_list, size=size, K=K, cos4a=cos4a)
# print(imgs.shape)
# semi_n.save_current_result = save_current_result
# K[:2,:] /= size

# t1 = time.time()
# rho, N= semi_n.main_loop(imgs, K, light_p, light_d, z0, mask0=None, mu=0, imgs_save_path=imgs_save_path)
# t2 = time.time()
# cv2.imwrite(imgs_save_path+'albedo.png', rho)
# semi_n.write_N_png(N, imgs_save_path+'N.png')
# print((t2-t1)/60, imgs.shape)
################################################ 

################################################ test0711
# size = 1                                                                                                                    # 下采样比例
# z0 = 628
# scales = 4                                                                                                                    # 场景平均高度
# K_path = '/home/ecoplants/hdd/changfa/PBR/semi_non_lbtm/data/rotatelights_resize.mat'                                                                                        # 相机内参
# lights_path = '/home/ecoplants/hdd/changfa/PBR/semi_non_lbtm/data/light_rotatelights.mat'                                                                                    # 相机坐标系下的光源坐标
# imgs_path = '/home/ecoplants/hdd/changfa/PBR/semi_non_lbtm/data/rotatelights/'                                                                                            # 图片文件夹
# imgs_save_path = 'out/changfa/new2/'                                                                                             # 结果保存路径
# save_current_result = True                                                                                                  # 是否保存中间结果
# cos4a=True                                                                                                                  # 原始图像是否进行cos4a的矫正

# semi_n = Semi_calib('data/config_opt.json')
# semi_n.scales=scales
# K = sio.loadmat(K_path)['K']
# # light_p = sio.loadmat(lights_path)['S']
# light_p = sio.loadmat(lights_path)['S']
# light_d = light_p - np.array([[0,0,z0]]); light_d = light_d / np.linalg.norm(light_d, axis=1)[:,None]
# # good_list = [0,1,2,3,4,5,6,7,10,11,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
# # light_p = light_p[good_list,:]; light_d = light_d[good_list,:]
# imgs_dir = os.listdir(imgs_path); imgs_dir.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))      # 注意图片名称
# imgs_dir_list = [imgs_path+x for x in imgs_dir]
# imgs = semi_n.load_imgs(imgs_dir_list, size=size, K=K, cos4a=cos4a)
# print(imgs.shape)
# semi_n.save_current_result = save_current_result
# K[:2,:] /= size

# t1 = time.time()
# rho, N= semi_n.main_loop(imgs, K, light_p, light_d, z0, mask0=None, mu=0, imgs_save_path=imgs_save_path)
# t2 = time.time()
# cv2.imwrite(imgs_save_path+'albedo.png', rho)
# semi_n.write_N_png(N, imgs_save_path+'N.png')
# print((t2-t1)/60, imgs.shape)

################################################## 材质扫描仪
# size = 8                                                                                                                    # 下采样比例
# z0 = 1070
# scales = 2                                                                                                                    # 场景平均高度
# K_path = 'testData/zmq/camera_ours_3.mat'                                                                                        # 相机内参
# lights_path = 'testData/zmq/light_ours.mat'                                                                                    # 相机坐标系下的光源坐标
# imgs_path = 'testData/zmq/'                                                                                            # 图片文件夹
# imgs_save_path = 'out/changfa/new3/'                                                                                             # 结果保存路径
# save_current_result = True                                                                                                  # 是否保存中间结果
# cos4a=True                                                                                                                  # 原始图像是否进行cos4a的矫正

# semi_n = Semi_calib('data/config_opt.json')
# semi_n.scales=scales
# K = sio.loadmat(K_path)['K']
# # light_p = sio.loadmat(lights_path)['S']
# light_p = sio.loadmat(lights_path)['S'][1:,:]
# light_d = light_p - np.array([[0,0,z0]]); light_d = light_d / np.linalg.norm(light_d, axis=1)[:,None]
# # good_list = [0,1,2,3,4,5,6,7,10,11,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
# # light_p = light_p[good_list,:]; light_d = light_d[good_list,:]
# imgs_dir = os.listdir(imgs_path); imgs_dir = [x for x in imgs_dir if x[-3:]=='png']
# imgs_dir.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))      # 注意图片名称
# # print(imgs_dir)
# imgs_dir_list = [imgs_path+x for x in imgs_dir]
# # for x in imgs_dir_list:
# #     print(x, cv2.imread(x,-1).shape)

# imgs = semi_n.load_imgs(imgs_dir_list, size=size, K=K, cos4a=cos4a)
# print(imgs.shape)
# semi_n.save_current_result = save_current_result
# K[:2,:] /= size

# if not os.path.exists(imgs_save_path):
#     os.makedirs(imgs_save_path)

# t1 = time.time()
# rho, N = semi_n.main_loop(imgs, K, light_p, light_d, z0, mask0=None, mu=0, imgs_save_path=imgs_save_path)
# t2 = time.time()
# cv2.imwrite(imgs_save_path+'albedo.png', rho)
# semi_n.write_N_png(N, imgs_save_path+'N.png')
# print((t2-t1)/60, imgs.shape)

##################################################################################
# size = 4                                                                                                                    # 下采样比例
# z0 = 730
# scales = 2                                                                                                                    # 场景平均高度
# K_path = 'testData/wutiplate/camera.mat'                                                                                        # 相机内参
# lights_path = 'testData/wutiplate/light_0726.mat'                                                                                    # 相机坐标系下的光源坐标
# imgs_path = 'testData/wutiplate/'                                                                                            # 图片文件夹
# imgs_save_path = 'out/changfa/new4/'                                                                                             # 结果保存路径
# save_current_result = True                                                                                                  # 是否保存中间结果
# cos4a=True                                                                                                                  # 原始图像是否进行cos4a的矫正

# semi_n = Semi_calib('data/config_opt.json')
# semi_n.scales=scales
# K = sio.loadmat(K_path)['K']
# light_p = sio.loadmat(lights_path)['S']
# # light_p = np.load(lights_path); light_p[:,1] = -light_p[:,1]; light_p[:,2] = z0 - light_p[:,2]
# light_d = light_p - np.array([[0,0,z0]]); light_d = light_d / np.linalg.norm(light_d, axis=1)[:,None]
# # good_list = [0,1,2,3,4,5,6,7,10,11,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
# # light_p = light_p[good_list,:]; light_d = light_d[good_list,:]
# imgs_dir = os.listdir(imgs_path); imgs_dir = [x for x in imgs_dir if x[-3:]=='png']
# imgs_dir.sort(key=lambda x:int(x[4:8]))      # 注意图片名称
# # print(imgs_dir)
# imgs_dir_list = [imgs_path+x for x in imgs_dir]
# imgs = semi_n.load_imgs(imgs_dir_list, size=size, K=K, cos4a=cos4a)
# print(imgs.shape)
# semi_n.save_current_result = save_current_result
# K[:2,:] /= size

# if not os.path.exists(imgs_save_path):
#     os.makedirs(imgs_save_path)

# t1 = time.time()
# rho, N= semi_n.main_loop(imgs, K, light_p, light_d, z0, mask0=None, mu=0, imgs_save_path=imgs_save_path)
# t2 = time.time()
# cv2.imwrite(imgs_save_path+'albedo.png', rho)
# semi_n.write_N_png(N, imgs_save_path+'N.png')
# print((t2-t1)/60, imgs.shape)

#####################################################################################
# size = 2                                                                                                                    # 下采样比例
# z0 = 730
# scales = 2                                                                                                                    # 场景平均高度
# K_path = 'testData/earphone/camera.mat'                                                                                        # 相机内参
# lights_path = 'testData/earphone/light.mat'                                                                                    # 相机坐标系下的光源坐标
# imgs_path = 'testData/earphone/'                                                                                            # 图片文件夹
# imgs_save_path = 'out/changfa/new5/'                                                                                             # 结果保存路径
# save_current_result = True                                                                                                  # 是否保存中间结果
# cos4a=True                                                                                                                  # 原始图像是否进行cos4a的矫正

# semi_n = Semi_calib('data/config_opt.json')
# semi_n.scales=scales
# K = sio.loadmat(K_path)['K']
# light_p = sio.loadmat(lights_path)['S']
# # light_p = np.load(lights_path); light_p[:,1] = -light_p[:,1]; light_p[:,2] = z0 - light_p[:,2]
# light_d = light_p - np.array([[0,0,z0]]); light_d = light_d / np.linalg.norm(light_d, axis=1)[:,None]
# # good_list = [0,1,2,3,4,5,6,7,10,11,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
# # light_p = light_p[good_list,:]; light_d = light_d[good_list,:]
# imgs_dir = os.listdir(imgs_path); imgs_dir = [x for x in imgs_dir if x[-3:]=='png']
# imgs_dir.sort(key=lambda x:int(x[4:8]))      # 注意图片名称
# # print(imgs_dir)
# imgs_dir_list = [imgs_path+x for x in imgs_dir]
# imgs = semi_n.load_imgs(imgs_dir_list, size=size, K=K, cos4a=cos4a)
# print(imgs.shape)
# semi_n.save_current_result = save_current_result
# K[:2,:] /= size

# if not os.path.exists(imgs_save_path):
#     os.makedirs(imgs_save_path)

# t1 = time.time()
# rho, N= semi_n.main_loop(imgs, K, light_p, light_d, z0, mask0=None, mu=0, imgs_save_path=imgs_save_path)
# t2 = time.time()
# cv2.imwrite(imgs_save_path+'albedo.png', rho)
# semi_n.write_N_png(N, imgs_save_path+'N.png')
# print((t2-t1)/60, imgs.shape)

###################################################################################### yellow mid
# size = 8                                                                                                                    # 下采样比例
# z0 = 730
# scales = 2                                                                                                                    # 场景平均高度
# K_path = 'testData/spoon/camera.mat'                                                                                        # 相机内参
# lights_path = 'testData/0727/light_coord.npy'                                                                                    # 相机坐标系下的光源坐标
# imgs_path = 'testData/0727/png/'                                                                                            # 图片文件夹
# imgs_save_path = 'out/changfa/new_mid/'                                                                                             # 结果保存路径
# save_current_result = True                                                                                                  # 是否保存中间结果
# cos4a=True                                                                                                                  # 原始图像是否进行cos4a的矫正

# semi_n = Semi_calib('data/config_opt.json')
# semi_n.scales=scales
# K = sio.loadmat(K_path)['K']
# # light_p = sio.loadmat(lights_path)['S']
# light_p = np.load(lights_path); light_p[:,1] = -light_p[:,1]; light_p[:,2] = z0 - light_p[:,2]
# light_d = light_p - np.array([[0,0,z0]]); light_d = light_d / np.linalg.norm(light_d, axis=1)[:,None]
# good_list = [0,1,2,3,5,6,7,9,10,11,15,18,19,21,22,23,24,25,26,27,29,30,31]
# light_p = light_p[good_list,:]; light_d = light_d[good_list,:]
# imgs_dir = os.listdir(imgs_path); imgs_dir.sort(key = lambda x: x.split('_')[2])      # 注意图片名称
# imgs_dir_list = [imgs_path+x for x in imgs_dir]
# # imgs = semi_n.load_imgs(imgs_dir_list, size=size, K=K, cos4a=cos4a)
# # print(imgs.shape)
# semi_n.save_current_result = save_current_result
# # K[:2,:] /= size
# K[0,0] = 13195.7348993873; K[0,2] = 2190.34771000672
# K[1,1] = 13191.4649472086; K[0,2] = 2316.28597025803
# K[:2,:] /= size

# if not os.path.exists(imgs_save_path):
#     os.makedirs(imgs_save_path)

# imgs = np.load('testData/yellow/mid_3h.npy').transpose(3,0,1,2)[:,::size,::size,:]; print(imgs.shape)
# good_list = [0,1,2,3,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,23,24,25]
# imgs = imgs[good_list]

# t1 = time.time()
# rho, N= semi_n.main_loop(imgs, K, light_p, light_d, z0, mask0=None, mu=0, imgs_save_path=imgs_save_path)
# t2 = time.time()
# cv2.imwrite(imgs_save_path+'albedo.png', rho)
# semi_n.write_N_png(N, imgs_save_path+'N.png')
# print((t2-t1)/60, imgs.shape)


################################################################################################ yellow up
size = 8                                                                                                                    # 下采样比例
z0 = 730
scales = 2                                                                                                                    # 场景平均高度
K_path = 'testData/spoon/camera.mat'                                                                                        # 相机内参
lights_path = 'testData/0727/light_coord.npy'                                                                                    # 相机坐标系下的光源坐标
imgs_path = 'testData/0727/png/'                                                                                            # 图片文件夹
imgs_save_path = 'out/changfa/new_earphoneup/'                                                                                             # 结果保存路径
save_current_result = True                                                                                                  # 是否保存中间结果
cos4a=True                                                                                                                  # 原始图像是否进行cos4a的矫正

semi_n = Semi_calib('data/config_opt.json')
semi_n.scales=scales
K = sio.loadmat(K_path)['K']
# light_p = sio.loadmat(lights_path)['S']
light_p = np.load(lights_path); light_p[:,1] = -light_p[:,1]; light_p[:,2] = z0 - light_p[:,2]; light_p[:,0] -= 75; light_p[:,1] += 75; 
# light_p = np.load(lights_path); light_p[:,1] = -light_p[:,1]; light_p[:,2] = z0 - light_p[:,2]

light_d = light_p - np.array([[-75,75,z0]]); light_d = light_d / np.linalg.norm(light_d, axis=1)[:,None]
good_list = [0,1,2,3,5,6,7,9,10,11,15,18,19,21,22,23,24,25,26,27,29,30,31]
light_p = light_p[good_list,:]; light_d = light_d[good_list,:]
imgs_dir = os.listdir(imgs_path); imgs_dir.sort(key = lambda x: x.split('_')[2])      # 注意图片名称
imgs_dir_list = [imgs_path+x for x in imgs_dir]
# imgs = semi_n.load_imgs(imgs_dir_list, size=size, K=K, cos4a=cos4a)
# print(imgs.shape)
semi_n.save_current_result = save_current_result
# K[:2,:] /= size
K[0,0] = 13195.7348993873; K[0,2] = 3470.34771000672
K[1,1] = 13191.4649472086; K[1,2] = 892.285970258026
# K[0,0] = 13195.7348993873; K[0,2] = 2190.34771000672
# K[1,1] = 13191.4649472086; K[1,2] = 2316.28597025803
K[:2,:] /= size

if not os.path.exists(imgs_save_path):
    os.makedirs(imgs_save_path)
imgs = np.load('testData/white/earphone_up_3h.npy').transpose(3,0,1,2)[:,::size,::size,:]; print(imgs.shape)
# good_list = [0,1,2,3,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,23,24,25]
# imgs = imgs[good_list]

t1 = time.time()
rho, N= semi_n.main_loop(imgs, K, light_p, light_d, z0, mask0=None, mu=0, imgs_save_path=imgs_save_path)
t2 = time.time()
cv2.imwrite(imgs_save_path+'albedo.png', rho)
semi_n.write_N_png(N, imgs_save_path+'N.png')
print((t2-t1)/60, imgs.shape)







