from semi_calib_shadow_zmq import Semi_calib
from scipy.interpolate import griddata
from tqdm import trange, tqdm
from glob import glob
import numpy as np
import cv2


def get_full_depth(depth, mask_path, camera_height):
    '''
       填充深度图的空值
       xml_path: 点云文件路径
       camera_height: 相机高度
       node_name: 'depth_saved'
    '''
    mask = cv2.imread(mask_path,0)
    a = mask[mask<255]
    # print(a[a>0])
    # print(mask.shape, pts.shape, np.max(mask))
    depth[mask<1]=camera_height
    # print(np.min(depth), np.max(depth))
    # depth[depth<1]=920

    grid_x, grid_y = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    valid = np.stack((grid_x, grid_y, depth), axis=-1)[depth > 0]
    points = valid[..., :2]
    values = valid[..., 2]
    add_valid = np.stack((grid_x, grid_y, np.zeros(depth.shape)), axis=-1)
    add_valid[0, :, 2] = camera_height
    add_valid[-1, :, 2] = camera_height
    add_valid[:, 0, 2] = camera_height
    add_valid[:, -1, 2] = camera_height
    add_valid = add_valid[add_valid[..., 2] != 0]
    add_points = add_valid[..., :2]
    add_values = add_valid[..., 2]
    full_depth = griddata(np.concatenate((points, add_points)), np.concatenate((values, add_values)), (grid_x, grid_y), method='cubic')
    return full_depth


# npy转点云
def depth2xyz(depth_map,depth_cam_matrix,flatten=True,depth_scale=1):
    fx,fy = depth_cam_matrix[0,0],depth_cam_matrix[1,1]
    cx,cy = depth_cam_matrix[0,2],depth_cam_matrix[1,2]
    h,w=np.mgrid[0:depth_map.shape[0],0:depth_map.shape[1]]
    z=depth_map/depth_scale
    x=(w-cx)*z/fx
    y=(h-cy)*z/fy
    xyz=np.dstack((x,y,z)) if flatten==False else np.dstack((x,y,z)).reshape(-1,3)
    #xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz


def xyz2dep(pc,cam_matrix):
    CAM_WID, CAM_HGT = 3200, 2400  # 重投影到的深度图尺寸
    CAM_FX, CAM_FY = cam_matrix[0,0], cam_matrix[1,1]  # fx/fy
    CAM_CX, CAM_CY = cam_matrix[0,2], cam_matrix[1,2]  # cx/cy

    EPS = 1.0e-16

    # 加载点云数据

    # 滤除镜头后方的点
    valid = pc[:, 2] > EPS
    z = pc[valid, 2]

    # 点云反向映射到像素坐标位置
    u = np.round(pc[valid, 0] * CAM_FX / z + CAM_CX).astype(int)
    v = np.round(pc[valid, 1] * CAM_FY / z + CAM_CY).astype(int)

    # 滤除超出图像尺寸的无效像素
    valid = np.bitwise_and(np.bitwise_and((u >= 0), (u < CAM_WID)),
                        np.bitwise_and((v >= 0), (v < CAM_HGT)))
    u, v, z = u[valid], v[valid], z[valid]

    # 按距离填充生成深度图，近距离覆盖远距离
    img_z = np.full((CAM_HGT, CAM_WID), np.inf)
    for ui, vi, zi in zip(u, v, z):
        img_z[vi, ui] = min(img_z[vi, ui], zi)  # 近距离像素屏蔽远距离像素

    # 小洞和“透射”消除
    img_z_shift = np.array([img_z, \
                            np.roll(img_z, 1, axis=0), \
                            np.roll(img_z, -1, axis=0), \
                            np.roll(img_z, 1, axis=1), \
                            np.roll(img_z, -1, axis=1)])
    img_z = np.min(img_z_shift, axis=0)

    return img_z

# if __name__ == "__main__":
    # mid_depth=r"D:\duomu\stone\depth_saved.xml"
    # index=r"D:\duomu\stone\depth_index.xml"
    # mask_path=r"D:\near_ps\testData\mid_stone\mask.png"
    # # 2400*3200内参
    # mid_K=np.array([[9418.703187998373, 0, 1521.306217883809],[0, 9412.140291652486, 1170.777850327491],[0, 0, 1]])
    # side_K=np.array([[9448.911990834111, 0, 1549.144952076924],[0, 9453.736170305898, 1135.398772807657],[0, 0, 1]])
    # # 外参
    # side_RT=np.array([[9.7914491968986417e-01, 8.6376074297607117e-03, 2.0297935358902630e-01, -2.0455719492252425e+02],
    #                 [-9.4080534192927841e-03, 9.9995173550471128e-01, 2.8311114382465145e-03, 3.1530650483845091e+00],
    #                 [-2.0294510286377782e-01, -4.6817089834140123e-03, 9.7917892482661484e-01, 6.8136566511372834e+01],
    #                 [0,0,0,1]])

    # fs1 = cv2.FileStorage(mid_depth, cv2.FileStorage_READ)
    # mid_depth = np.array(fs1.getNode('depth_saved').mat())
    # mid_depth=get_full_depth(mid_depth,mask_path, np.mean(mid_depth[mid_depth>0]))

    # # 主相机点云
    # mid_cloud = depth2xyz(mid_depth, mid_K)

    # # np.save(r"D:\duomu\stone\mid_depth.npy",mid_depth)
    # # np.savetxt(r"D:\duomu\stone\mid_cloud.txt", mid_cloud)

    # fs2 = cv2.FileStorage(index, cv2.FileStorage_READ)
    # width = np.array(fs2.getNode("width").mat()).astype(np.int16)
    # height = np.array(fs2.getNode("height").mat()).astype(np.int16)

    # # 深度图大小
    # hw = [width.shape[0], width.shape[1]]

    # # 主相机点云转到侧相机坐标系
    # mid_temp=np.concatenate((mid_cloud.T,np.ones((1,mid_cloud.shape[0]))),axis=0)
    # side_cloud=np.matmul(side_RT,mid_temp).T[:,:3]
    # side_depth=np.reshape(side_cloud[:,2],(hw[0],hw[1]))

    # # x=np.matmul(side_RT,np.array([[0],[0],[0],[1]]))[2,:]
    # # side_depth[side_depth==x]=0

    # # 侧相机深度npy
    # side_new=np.zeros_like(hw)
    # #根据index对照侧相机深度图 
    # side_new[height[height!=-1],width[width!=-1]]=side_depth[height!=-1]

    # side_new=get_full_depth(side_new,mask_path,np.mean(side_new[side_new>0]))

    # # np.save(r"D:\duomu\stone\side_depth.npy",side_new)
    # # np.savetxt(r"D:\duomu\stone\side_cloud.txt",side_cloud)



    # mid_depth_new=Semi_calib('data/config_hzj.json').run(mid_depth)
    # side_depth_new=Semi_calib('data/config_hzj.json').run(side_new)

    #     # 半标定抛出的结果进行插值
    # if mid_depth_new.shape[0]!=mid_depth.shape[0]:
    #     mid_depth_new=cv2.resize(mid_depth_new,(mid_depth.shape[1],mid_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    #     side_depth_new=cv2.resize(side_depth_new,(mid_depth.shape[1],mid_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    # # np.save(r"D:\duomu\stone\side_depth.npy", mid_depth_new)
    # # np.save(r"D:\duomu\stone\side_depth.npy", side_depth_new)

    # mid_cloud_new = depth2xyz(mid_depth_new, mid_K)
    # side_cloud_new = depth2xyz(side_depth_new, side_K)

    # side_temp=np.concatenate((side_cloud_new.T,np.ones((1,side_cloud_new.shape[0]))),axis=0)
    # side_cloud_new = np.matmul(np.linalg.inv(side_RT), side_temp).T[:, :3]
    # np.savetxt(r"D:\near_ps\testData\side_stone\side_cloud4_RT.txt", side_cloud_new)
    # np.savetxt(r"D:\duomu\stone\side_cloud.txt", mid_cloud_new)


if __name__ == "__main__":

    mid_cloud_path=r""
    mask_path=r"D:\near_ps\testData\mid_stone\mask.png"
    mid_K=np.array([[9418.703187998373, 0, 1521.306217883809],[0, 9412.140291652486, 1170.777850327491],[0, 0, 1]])
    side_K=np.array([[9448.911990834111, 0, 1549.144952076924],[0, 9453.736170305898, 1135.398772807657],[0, 0, 1]])
    # 外参
    side_RT=np.array([[9.7914491968986417e-01, 8.6376074297607117e-03, 2.0297935358902630e-01, -2.0455719492252425e+02],
                    [-9.4080534192927841e-03, 9.9995173550471128e-01, 2.8311114382465145e-03, 3.1530650483845091e+00],
                    [-2.0294510286377782e-01, -4.6817089834140123e-03, 9.7917892482661484e-01, 6.8136566511372834e+01],
                    [0,0,0,1]])

    # 主相机深度图
    mid_cloud=np.loadtxt(mid_cloud_path)
    mid_dep=xyz2dep(mid_cloud,mid_K)
    mid_dep=get_full_depth(mid_dep,mask_path,mid_K)
    mid_cloud=depth2xyz(mid_dep,mid_K)

    # 侧相机深度图
    side_temp = np.concatenate((mid_cloud.T,np.ones((1,mid_cloud.shape[0]))),axis=0)                              #注意inf以及去噪，插值等
    side_cloud = np.matmul(side_RT, side_temp).T[:, :3]
    side_dep=xyz2dep(side_cloud,side_K)

    #光度立体生成新点云       
    mid_dep_new=Semi_calib('data/config_hzj.json').run(mid_dep)
    side_dep_new=Semi_calib('data/config_hzj.json').run(side_dep)

    mid_cloud_new=depth2xyz(mid_dep_new,mid_K)
    side_cloud_new=depth2xyz(side_dep_new,side_K)

    side_temp=np.concatenate((side_cloud_new.T,np.ones((1,side_cloud_new.shape[0]))),axis=0)
    side_cloud_RT = np.matmul(np.linalg.inv(side_RT), side_temp).T[:, :3]

    