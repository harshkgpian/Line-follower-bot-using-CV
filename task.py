from cvzone.ColorModule import ColorFinder
import cv2
import numpy as np
import os, sys
import traceback
import math
import time
import sys


##############################################################


# Importing the sim module for Remote API connection with CoppeliaSim
try:
	import sim
	
except Exception:
	print('\n[ERROR] It seems the sim.py OR simConst.py files are not found!')
	print('\n[WARNING] Make sure to have following files in the directory:')
	print('sim.py, simConst.py and appropriate library - remoteApi.dll (if on Windows), remoteApi.so (if on Linux) or remoteApi.dylib (if on Mac).\n')
	sys.exit()



def init_remote_api_server():
    client_id =-1    
    sim.simxFinish(-1)
    client_id = sim.simxStart('127.0.0.1',19997,True,True,5000,5)
    return client_id


def start_simulation(client_id):
	return_code = -2
	if client_id != -1 :
		return_code = sim.simxStartSimulation(client_id,sim.simx_opmode_blocking)

	return return_code






def stop_simulation(client_id):
	return_code = -2
	return_code = sim.simxStopSimulation(client_id, sim.simx_opmode_blocking)
	return return_code


def exit_remote_api_server(client_id):
	sim.simxFinish(client_id)	



def set_bot_movement(client_id,wheel_joints,forw_back_vel,left_right_vel,rot_vel):
    sim.simxSetJointTargetVelocity(client_id,wheel_joints[0],-forw_back_vel-left_right_vel-rot_vel,sim.simx_opmode_blocking)
    sim.simxSetJointTargetVelocity(client_id,wheel_joints[1],-forw_back_vel+left_right_vel-rot_vel,sim.simx_opmode_blocking)
    sim.simxSetJointTargetVelocity(client_id,wheel_joints[2],-forw_back_vel-left_right_vel+rot_vel,sim.simx_opmode_blocking)
    sim.simxSetJointTargetVelocity(client_id,wheel_joints[3],-forw_back_vel+left_right_vel+rot_vel,sim.simx_opmode_blocking)


def init_setup(client_id):
	wheel_joints=[-1,-1,-1,-1]   ##front left, rear left, rear right, front right
	return_code, wheel_joints[0]=sim.simxGetObjectHandle(client_id, 'rollingJoint_fl',sim.simx_opmode_blocking)
	return_code,wheel_joints[1]=sim.simxGetObjectHandle(client_id,'rollingJoint_rl',sim.simx_opmode_blocking)
	return_code,wheel_joints[2]=sim.simxGetObjectHandle(client_id,'rollingJoint_rr',sim.simx_opmode_blocking)
	return_code,wheel_joints[3]=sim.simxGetObjectHandle(client_id,'rollingJoint_fr',sim.simx_opmode_blocking)
	return wheel_joints


def encoders(client_id):
	return_code,signal_value=sim.simxGetStringSignal(client_id,'combined_joint_position',sim.simx_opmode_blocking)
	signal_value = signal_value.decode()
	joints_position = signal_value.split("%")

	for index,joint_val in enumerate(joints_position):
		joints_position[index]=float(joint_val)

	return joints_position

def task_primary(client_id):
	image_data(client_id)
	# play(client_id)
	

# image processing
def image_data(client_id):
	a = 0
	b = 0
	#thresholvalues
	hsvVals = {'hmin': 0, 'smin': 0, 'vmin': 230, 'hmax': 179, 'smax': 255, 'vmax': 255} 
    # getting data from coppeliasim camera
	res, v1 = sim.simxGetObjectHandle(client_id, 'camera', sim.simx_opmode_oneshot_wait)
	err, resolution, image = sim.simxGetVisionSensorImage(client_id, v1, 0, sim.simx_opmode_streaming)
	img = np.array(image,dtype=np.uint8)
	img.resize(256,256,3)
	img = cv2.flip(img,0)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # circular mask
	circle= np.zeros((img.shape[0],img.shape[1]), dtype="uint8")
	cir = cv2.circle(circle,(img.shape[0]//2,img.shape[1]+120),img.shape[1]-70,(255,255,255),10)
	
	while (sim.simxGetConnectionId(client_id) != -1):
		err, resolution, image = sim.simxGetVisionSensorImage(client_id, v1, 0, sim.simx_opmode_buffer)
		if err == sim.simx_return_ok:
			img = np.array(image,dtype=np.uint8)	
			img.resize([resolution[1],resolution[0],3])
			img = cv2.flip(img,0)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			imgColor , mask = threaholding(hsvVals,img)
			
            # masking path and circular arc
			I = cv2.bitwise_and(mask,cir)
			biggest, cx, cy = draw_contours(I)
			cv2.drawContours(img,biggest,-1,(255,0,255),7)
			cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)
			x = I.shape[0]//2
			y = I.shape[1]
            # heading angle
			w_rot = np.arctan((x-cx)/(y-cy))
			cv2.circle(I,(I.shape[0]//2,I.shape[1]),10,(255,0,0),cv2.FILLED)
			cv2.imshow('ima',img)
			cv2.imshow('image',I)
			a= abs(np.rad2deg(w_rot))
			if a<=30:
				n = 7
				m = 0.5
			elif a>20 and a<=50:
				n = 5
				m = 1.2
			elif a>50 and a<=70:
				n = 4
				m = 1.8
			else:
				m = 0.8
				n= 2
			print(a)
			set_bot_movement(client_id,init_setup(client_id),-n,0,m*w_rot)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		elif err == sim.simx_return_novalue_flag:
			# print("no image yet")
			pass
		else:
			print(err)

	cv2.destroyAllWindows()


# to segment the red path
def threaholding(hsvVals,img):
	myColorFinder = ColorFinder(False)
	imgColor, mask = myColorFinder.update(img, hsvVals)
	return imgColor , mask

# returns the centre of the biggest contour found
def draw_contours(mask):
	contours , hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	biggest = max(contours, key = cv2.contourArea)
	x,y,w,h = cv2.boundingRect(biggest)
	cx = x+ w//2
	cy = y + h//2
	return biggest,cx,cy



if __name__ == "__main__":

	print('\nConnection to CoppeliaSim Remote API Server initiated.')
	print('Trying to connect to Remote API Server...')

	try:
		client_id = init_remote_api_server()
		if (client_id != -1):
			print('\nConnected successfully to Remote API Server in CoppeliaSim!')

			# Starting the Simulation
			try:
				return_code = start_simulation(client_id)

				if (return_code == sim.simx_return_novalue_flag) or (return_code == sim.simx_return_ok):
					print('\nSimulation started correctly in CoppeliaSim.')

				else:
					print('\n[ERROR] Failed starting the simulation in CoppeliaSim!')
					print('start_simulation function is not configured correctly, check the code!')
					print()
					sys.exit()

			except Exception:
				print('\n[ERROR] Your start_simulation function throwed an Exception, kindly debug your code!')
				print('Stop the CoppeliaSim simulation manually.\n')
				traceback.print_exc(file=sys.stdout)
				print()
				sys.exit()

		else:
			print('\n[ERROR] Failed connecting to Remote API server!')
			print('[WARNING] Make sure the CoppeliaSim software is running and')
			print('[WARNING] Make sure the Port number for Remote API Server is set to 19997.')
			print('[ERROR] OR init_remote_api_server function is not configured correctly, check the code!')
			print()
			sys.exit()

	except Exception:
		print('\n[ERROR] Your init_remote_api_server function throwed an Exception, kindly debug your code!')
		print('Stop the CoppeliaSim simulation manually if started.\n')
		traceback.print_exc(file=sys.stdout)
		print()
		sys.exit()

	try:

		task_primary(client_id)
		time.sleep(2)        

		try:
			return_code = stop_simulation(client_id)                            
			if (return_code == sim.simx_return_ok) or (return_code == sim.simx_return_novalue_flag):
				print('\nSimulation stopped correctly.')

				# Stop the Remote API connection with CoppeliaSim server
				try:
					exit_remote_api_server(client_id)
					if (start_simulation(client_id) == sim.simx_return_initialize_error_flag):
						print('\nDisconnected successfully from Remote API Server in CoppeliaSim!')

					else:
						print('\n[ERROR] Failed disconnecting from Remote API server!')
						print('[ERROR] exit_remote_api_server function is not configured correctly, check the code!')

				except Exception:
					print('\n[ERROR] Your exit_remote_api_server function throwed an Exception, kindly debug your code!')
					print('Stop the CoppeliaSim simulation manually.\n')
					traceback.print_exc(file=sys.stdout)
					print()
					sys.exit()
									  
			else:
				print('\n[ERROR] Failed stopping the simulation in CoppeliaSim server!')
				print('[ERROR] stop_simulation function is not configured correctly, check the code!')
				print('Stop the CoppeliaSim simulation manually.')
		  
			print()
			sys.exit()

		except Exception:
			print('\n[ERROR] Your stop_simulation function throwed an Exception, kindly debug your code!')
			print('Stop the CoppeliaSim simulation manually.\n')
			traceback.print_exc(file=sys.stdout)
			print()
			sys.exit()

	except Exception:
		print('\n[ERROR] Your task_3_primary function throwed an Exception, kindly debug your code!')
		print('Stop the CoppeliaSim simulation manually if started.\n')
		traceback.print_exc(file=sys.stdout)
		print()
		sys.exit()
