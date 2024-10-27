#!/usr/bin/env python
# license removed for brevity
import math
import rospy
import message_filters
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import numpy as np
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R

transform_LidarToCam = np.array([
    [0.998542,  -0.00512128,  -0.0537304,-0.0505882],
    [0.00568941 , 0.999929,  0.0104262,-0.261617],
    [0.0536733,  -0.0107167 , 0.998501,-0.0290247],
    [0 , 0,  0  ,1],
])

transform_CamToExcavator = np.array([
    [0.82903757,  0, 0.5591929, 0.38,],
    [0, 1, 0,0.95],
    [-0.5591929,  0, 0.82903757,2.455],
    [0 , 0,  0  ,1],
])

transform_CamZToCamX = np.array([
    [0,  0, 1,  0,],
    [-1,  0,  0,  0,],
    [0,  -1,  0,  0,],
    [0,  0,  0,  1,],
])


def ToQuaternion(yaw_, pitch_, roll_):
    yaw =math.radians(yaw_)
    pitch =math.radians(pitch_)
    roll =math.radians(roll_)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr =  math.sin(roll * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return x,y,z,w

def ToEulerian(x,y,z,w):
    roll = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch = math.asin(2*(w*y-z*x))
    yaw = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))
    return roll,pitch,yaw

def quatProduct(q1,q2):
    r1 = q1[3]
    r2 = q2[3]
    v1 = np.array([q1[0], q1[1], q1[2]])
    v2 = np.array([q2[0], q2[1], q2[2]])

    r = r1 * r2 - np.dot(v1, v2)
    v = r1 * v2 + r2 * v1 + np.cross(v1, v2)
    q = np.array([v[0], v[1], v[2],r])
    return q

def transform_concatenate(rotation_matrix,translation_vec):
    translation_vec_ = translation_vec.reshape(3,1)
    basic_vec = np.array([0,0,0,1]).reshape(1,4)
    transform_matrix34 = np.concatenate((rotation_matrix,translation_vec_),axis=1)
    transform_matrix44 = np.concatenate((transform_matrix34,basic_vec),axis=0)
    return transform_matrix44

def transformToRT(transform_matrix44):
    rotation_matrix = transform_matrix44[0:3,0:3]
    translation_vec = transform_matrix44[0:3,3:]
    return rotation_matrix, translation_vec

def CallBack1(odom):
    # roll,pitch,yaw = ToEulerian(odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)
    rotate1 = np.array([odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w])
    translation = np.array([odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z])
    poseR = R.from_quat(rotate1)
    pose_matrix = transform_concatenate(poseR.as_dcm(),translation)
    pose_cam = np.dot(transform_LidarToCam,pose_matrix)
    pose_excavator = np.dot(transform_CamToExcavator,pose_cam)
    rotation_matrix,tranlation_vec = transformToRT(pose_excavator)
    rotation = R.from_matrix(rotation_matrix)
    print(" rotation",rotation.as_euler('zyx',degrees=True))
    print("translation",tranlation_vec.reshape(1,3))
    rate.sleep()

def CallBackORBSLAM2AndLidar(odomORB,odomLidar):
    #CAM raw rotation
    CAM_raw_rotation_quat = np.array([odomORB.pose.pose.orientation.x,odomORB.pose.pose.orientation.y,odomORB.pose.pose.orientation.z,odomORB.pose.pose.orientation.w])
    CAM_raw_rotation = R.from_quat(CAM_raw_rotation_quat)
    CAM_raw_rotation_matrix = CAM_raw_rotation.as_dcm()#python3 use as_matrix()
    CAM_raw_rotation_euler = CAM_raw_rotation.as_euler('zyx',degrees=True)
    #CAM raw translation
    CAM_raw_translation13 = np.array([odomORB.pose.pose.position.x,odomORB.pose.pose.position.y,odomORB.pose.pose.position.z])
    CAM_raw_translation13mm = 1000*CAM_raw_translation13
    # print("RAW Translation:",CAM_raw_translation13mm)
    # print("RAW Rotation      :",CAM_raw_rotation_euler)
    #CAM raw transform
    CAM_raw_transform_matrix = transform_concatenate(CAM_raw_rotation_matrix,CAM_raw_translation13)
    #camera coordinate system transform
    R_CAM ,_=transformToRT(transform_CamZToCamX)
    CAM_rotation_matrix = (R_CAM.dot(CAM_raw_rotation_matrix)).dot(np.linalg.inv(R_CAM))
    CAM_translation31 = R_CAM.dot(CAM_raw_translation13.reshape(3,1))
    CAM_rotation = R.from_dcm(CAM_rotation_matrix)
    CAM_rotation_quat = CAM_rotation.as_quat()
    CAM_rotation_euler = CAM_rotation.as_euler('zyx',degrees=True)
    #CAM  translation
    CAM_translation13 = CAM_translation31.reshape(1,3)

    #Lidar rotation
    Lidar_rotation_quat = np.array([odomLidar.pose.pose.orientation.x,odomLidar.pose.pose.orientation.y,odomLidar.pose.pose.orientation.z,odomLidar.pose.pose.orientation.w])
    Lidar_rotation = R.from_quat(Lidar_rotation_quat)
    Lidar_rotation_matrix = Lidar_rotation.as_dcm()#python3 use as_matrix()
    Lidar_rotation_euler = Lidar_rotation.as_euler('zyx',degrees=True)

    #Excavator rotation
    #From CAM
    R_CToE , t_CToE=transformToRT(transform_CamToExcavator)
    Ec_rotation_matrix = (R_CToE.dot(CAM_rotation_matrix)).dot(R_CToE.T)
    Ec_rotation = R.from_dcm(Ec_rotation_matrix)
    Ec_rotation_euler = Ec_rotation.as_euler('zyx',degrees=True)
    #From Lidar
    x,y,z,w = ToQuaternion(0,31,0)
    EL_rotation = R.from_quat(quatProduct((quatProduct(np.array([x,y,z,w]),Lidar_rotation_quat)),np.array([-x,-y,-z,w])))
    EL_rotation_euler = EL_rotation.as_euler('zyx',degrees=True)
    EL_rotation_matrix = EL_rotation.as_dcm()
    print('excavator_rotation:' ,excavator_translation13mm)

    #Excavator Translation
    #From CAM
    t_CToE_realtime =  Ec_rotation_matrix.dot(t_CToE)
    excavator_translation31 = R_CToE.dot(CAM_translation31) + t_CToE - t_CToE_realtime
    excavator_translation13 = excavator_translation31.reshape(1,3)
    excavator_translation13mm = 1000*excavator_translation13
    print('excavator_translation:' ,excavator_translation13mm)

    array = [ EL_rotation_euler[2], EL_rotation_euler[1], EL_rotation_euler[0],excavator_translation13mm[0][0],excavator_translation13mm[0][1],excavator_translation13mm[0][2]]
    rotation = Float64MultiArray(data=array)
    pub.publish(rotation)

def CallBackORBSLAM2(odom):
    #sensor raw rotation
    sensor_raw_rotation_quat = np.array([odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w])
    sensor_raw_rotation = R.from_quat(sensor_raw_rotation_quat)
    sensor_raw_rotation_matrix = sensor_raw_rotation.as_dcm()#python3 use as_matrix()
    sensor_raw_rotation_euler = sensor_raw_rotation.as_euler('zyx',degrees=True)
    #sensor raw translation
    sensor_raw_translation13 = np.array([odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z])
    sensor_raw_translation13mm = 1000*sensor_raw_translation13
    # print("RAW Translation:",sensor_raw_translation13mm)
    # print("RAW Rotation      :",sensor_raw_rotation_euler)
    #sensor raw transform
    sensor_raw_transform_matrix = transform_concatenate(sensor_raw_rotation_matrix,sensor_raw_translation13)
    #camera coordinate system transform
    R_CAM ,_=transformToRT(transform_CamZToCamX)
    sensor_rotation_matrix = (R_CAM.dot(sensor_raw_rotation_matrix)).dot(np.linalg.inv(R_CAM))
    sensor_translation31 = R_CAM.dot(sensor_raw_translation13.reshape(3,1))
    sensor_rotation = R.from_dcm(sensor_rotation_matrix)
    sensor_rotation_quat = sensor_rotation.as_quat()
    sensor_rotation_euler = sensor_rotation.as_euler('zyx',degrees=True)
    #sensor  translation
    sensor_translation13 = sensor_translation31.reshape(1,3)
    print("Translation:",1000*sensor_translation13)
    print("Rotation      :",sensor_rotation_euler)

    #lidar transform rotation direct
    x,y,z,w = ToQuaternion(0,34,0)
    excavator_rotation_refer = R.from_quat(quatProduct((quatProduct(np.array([x,y,z,w]),sensor_rotation_quat)),np.array([-x,-y,-z,w])))
    excavator_rotation_refer_euler = excavator_rotation_refer.as_euler('zyx',degrees=True)
    excavator_rotation_refer_matrix = excavator_rotation_refer.as_dcm()
    #transform matrixs
    R_LToE , t_LToE=transformToRT(transform_CamToExcavator)
    #excavator transform rotation
    excavator_rotation_matrix = (R_LToE.dot(sensor_rotation_matrix)).dot(R_LToE.T)
    excavator_rotation = R.from_dcm(excavator_rotation_matrix)
    excavator_rotation_euler = excavator_rotation.as_euler('zyx',degrees=True)
    print("excavator_rotation_refers:", excavator_rotation_refer_euler)
    print("excavator_rotation_euler:",excavator_rotation_euler)
    # print("excavator_rotation_erro:",excavator_rotation_euler- excavator_rotation_refer_euler)
    #excavator transform translation
    t_LToE_realtime =  excavator_rotation_matrix.dot(t_LToE)
    excavator_translation31 = R_LToE.dot(sensor_translation13.reshape(3,1)) + t_LToE - t_LToE_realtime
    excavator_translation13 = excavator_translation31.reshape(1,3)
    excavator_translation13mm = 1000*excavator_translation13
    print("excavator_translation:" ,excavator_translation13mm)
    array = [excavator_rotation_euler[2],excavator_rotation_euler[1],excavator_rotation_euler[0],excavator_translation13mm[0][0],excavator_translation13mm[0][1],excavator_translation13mm[0][2]]
    rotation = Float64MultiArray(data=array)
    pub.publish(rotation)

    print("\n")

def CallBackLidar(odom):
    #sensor rotation
    sensor_rotation_quat = np.array([odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w])
    sensor_rotation = R.from_quat(sensor_rotation_quat)
    sensor_rotation_matrix = sensor_rotation.as_dcm()#python3 use as_matrix()
    sensor_rotation_euler = sensor_rotation.as_euler('zyx',degrees=True)
    #sensor translation
    sensor_translation13 = np.array([odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z])
    sensor_translation13mm = 1000*sensor_translation13
    print("Translation:",sensor_translation13mm)
    print("Rotation      :",sensor_rotation_euler)
    #lidar transform rotation direct
    x,y,z,w = ToQuaternion(0,31,0)
    excavator_rotation_refer = R.from_quat(quatProduct((quatProduct(np.array([x,y,z,w]),sensor_rotation_quat)),np.array([-x,-y,-z,w])))
    excavator_rotation_refer_euler = excavator_rotation_refer.as_euler('zyx',degrees=True)
    excavator_rotation_refer_matrix = excavator_rotation_refer.as_dcm()
    #transform matrixs
    R_LToE , t_LToE=transformToRT(transform_CamToExcavator.dot(transform_LidarToCam))
    #excavator transform rotation
    excavator_rotation_matrix = (R_LToE.dot(sensor_rotation_matrix)).dot(R_LToE.T)
    excavator_rotation = R.from_dcm(excavator_rotation_matrix)
    excavator_rotation_euler = excavator_rotation.as_euler('zyx',degrees=True)
    print("excavator_rotation_refers:", excavator_rotation_refer_euler)
    print("excavator_rotation_euler:",excavator_rotation_euler)
    # print("excavator_rotation_erro:",excavator_rotation_euler- excavator_rotation_refer_euler)
    #excavator transform translation
    t_LToE_realtime =  excavator_rotation_matrix.dot(t_LToE)
    excavator_translation31 = R_LToE.dot(sensor_translation13.reshape(3,1)) + t_LToE - t_LToE_realtime
    excavator_translation13 = excavator_translation31.reshape(1,3)
    excavator_translation13mm = 1000*excavator_translation13
    print("excavator_translation:" ,excavator_translation13mm)
    array = [excavator_rotation_euler[2],excavator_rotation_euler[1],excavator_rotation_euler[0],excavator_translation13mm[0][0],excavator_translation13mm[0][1],excavator_translation13mm[0][2]]
    rotation = Float64MultiArray(data=array)
    pub.publish(rotation)
    print("\n")




def CallBack2(odom):
    rotate1 = np.array([odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w])
    translation = np.array([odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z])
    poseR = R.from_quat(rotate1)
    RT = poseR.as_dcm()
    translation = RT.T.dot(translation.reshape(3,1))

    pose_matrix = transform_concatenate(poseR.as_dcm(),translation)
    # pose_matrix = np.dot(transform_LidarToCam,pose_matrix)
    # pose_matrix = np.dot(transform_CamToExcavator,pose_matrix)
    rotation_matrix,tranlation_vec = transformToRT(pose_matrix)
    pose_excavator_rotation = R.from_matrix(rotation_matrix)
    pose_excavator_rotation_euler = pose_excavator_rotation.as_euler('zyx',degrees=True)
    tranlation_vec = tranlation_vec.reshape(1,3)
    tranlation_vec = 1000*tranlation_vec
    print(" rotation",pose_excavator_rotation_euler)
    print("translation",tranlation_vec)
    x,y,z,w = ToQuaternion(0,31,0)
    rotate2 = np.array([x,y,z,w])
    rotate = quatProduct(rotate2,rotate1)
    roll,pitch,yaw = ToEulerian(rotate[0],rotate[1],rotate[2],rotate[3])
    array = [math.degrees(roll),math.degrees(pitch),math.degrees(yaw),tranlation_vec[0][0],tranlation_vec[0][1],tranlation_vec[0][2]]
    rotation = Float64MultiArray(data=array)
    pub.publish(rotation)
    print("roll(x): ",math.degrees(roll))
    print("pitch(y): ",math.degrees(pitch))
    print("yaw(z): ",math.degrees(yaw))
    print("rotation(z) -  yaw(z)= ",pose_excavator_rotation_euler[0] - math.degrees(yaw))
    rate.sleep()

def CallBack(odom):
    rotate1 = np.array([odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w])
    translation = np.array([odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z])
    poseR = R.from_quat(rotate1)
    pose_matrix = transform_concatenate(poseR.as_dcm(),translation)
    transform_LidarToExcavator = transform_CamToExcavator.dot(transform_LidarToCam)
    pose_matrix = np.dot(np.dot(transform_LidarToExcavator,pose_matrix),np.linalg.inv(transform_LidarToExcavator))
    rotation_matrix,tranlation_vec = transformToRT(pose_matrix)
    pose_excavator_rotation = R.from_dcm(rotation_matrix)
    pose_excavator_rotation_euler = pose_excavator_rotation.as_euler('zyx',degrees=True)
    tranlation_vec = tranlation_vec.reshape(1,3)
    tranlation_vec = 1000*tranlation_vec
    print(" rotation",pose_excavator_rotation_euler)
    print("translation",tranlation_vec)
    x,y,z,w = ToQuaternion(0,31,0)
    rotate2 = np.array([x,y,z,w])
    rotate = quatProduct(rotate2,rotate1)
    roll,pitch,yaw = ToEulerian(rotate[0],rotate[1],rotate[2],rotate[3])
    array = [math.degrees(roll),math.degrees(pitch),math.degrees(yaw),tranlation_vec[0][0],tranlation_vec[0][1],tranlation_vec[0][2]]
    rotation = Float64MultiArray(data=array)
    pub.publish(rotation)
    print("roll(x): ",math.degrees(roll))
    print("pitch(y): ",math.degrees(pitch))
    print("yaw(z): ",math.degrees(yaw))
    print("rotation(z) -  yaw(z)= ",pose_excavator_rotation_euler[0] - math.degrees(yaw))
    rate.sleep()

def ros_node():
    print("odom_to_angles Init")
    rospy.init_node('odom_to_angles', anonymous=True)
    rospy.Subscriber("/orbslam2_odom", Odometry,CallBackORBSLAM2)
    # rospy.Subscriber("/Odometry", Odometry, CallBackLidar)
    rospy.Subscriber("/Odometry", Odometry, CallBack)
    #Lidar and Cam subscribe
    # sub_Cam = message_filters.Subscriber("/orbslam2_odom" , Odometry)
    # sub_Lidar = message_filters.Subscriber("/Odometry", Odometry)
    # ts = message_filters.ApproximateTimeSynchronizer([sub_Cam , sub_Lidar] ,1 , 0.1,allow_headerless=True)
    # ts.registerCallback(CallBackORBSLAM2AndLidar)

    global pub
    pub = rospy.Publisher("/rotation",Float64MultiArray, queue_size=10)
    global rate
    rate = rospy.Rate(50)
    rospy.spin()

if __name__ == '__main__':
    ros_node()
    # R_LToE , t_LToE=transformToRT(transform_CamToExcavator.dot(transform_LidarToCam))
    # r = R.from_dcm(R_LToE)
    # print(t_LToE.reshape(1,3))
    # print(r.as_euler('zyx',degrees=True))


