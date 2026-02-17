import socket
import cv2
import mediapipe as mp
import json
import time
import matplotlib.pyplot as plt
import numpy as np 
import os

# Configuration constants
TARGET_FPS = 60
FRAME_TIME = 1 / TARGET_FPS
last_time = 0

# Stores coordinates for the graph
right_hand_path_x = []
right_hand_path_y = []
left_hand_path_x = []
left_hand_path_y = []

# Stores the last valid position of each hand
last_known_right_hand = None
last_known_left_hand = None

# Lists for the moving average filter
right_hand_history = []
left_hand_history = []
FILTER_SIZE = 5

# Variables to store session metadata
patient_name = "Anonimo"
date_str = "Data_Desconhecida"
time_str = "Hora_Desconhecida"
save_path = ""

#############################################################################
#                           TCP Client Configuration                        #
#############################################################################
HOST = '127.0.0.1'
PORT = 12345
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#############################################################################

try:
	client_socket.connect((HOST, PORT))
	print("Successfully connected to the Godot server!")

	# 1. RECEIVE METADATA FROM GODOT (Name, Date, Time)
	print("Waiting for patient metadata...")
	metadata_json = client_socket.makefile().readline().strip()
	
	if metadata_json:
		metadata = json.loads(metadata_json)
		
		# Extract data
		patient_name = metadata.get("name", "Anonimo")
		date_str = metadata.get("date", "Data_Desconhecida")
		time_str = metadata.get("time", "Hora_Desconhecida")
		sanitized_time_str = time_str.replace(':', '-')
		
		print(f"Metadata received: {patient_name} at {date_str} {time_str}")
		save_dir = "relatorios_pacientes"
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		
		# Create filename and path
		filename = f"{patient_name.replace(' ', '_')}_{date_str}_{sanitized_time_str}.png"
		save_path = os.path.join(save_dir, filename)


#############################################################################
#                         Video Capture Configuration                       #
#############################################################################

	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print("The script is running. Press Ctrl+C to stop.")
	
#############################################################################

	# Initialize MediaPipe Hands
	mp_hands = mp.solutions.hands
	
	# Indices of hand landmarks to be sent
	pontos_mao = [0, 4, 8, 9, 12, 16, 20]

	with mp_hands.Hands(
		max_num_hands=2,
		min_detection_confidence=0.7,
		min_tracking_confidence=0.3
	) as hands:
		while cap.isOpened():
			current_time_loop = time.time()
			if (current_time_loop - last_time) < FRAME_TIME:
				continue
			last_time = current_time_loop

			ret, frame = cap.read()
			if not ret:
				break

			# Frame processing for hand detection
			frame = cv2.flip(frame, 1)
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image.flags.writeable = False
			results = hands.process(image)
			image.flags.writeable = True

			current_detected_hands = {"Right": False, "Left": False}
			hands_data = {}

			if results.multi_hand_landmarks:
				for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
					hand_label = handedness.classification[0].label
					current_detected_hands[hand_label] = True
					
					landmarks = [
						{
							"index": idx,
							"x": hand_landmarks.landmark[idx].x,
							"y": hand_landmarks.landmark[idx].y,
							"z": hand_landmarks.landmark[idx].z
						}
						for idx in pontos_mao
					]
					
					# Apply moving average filter
					if hand_label == "Right":
						right_hand_history.append(landmarks)
						if len(right_hand_history) > FILTER_SIZE:
							right_hand_history.pop(0)
						
						# Calculate the average of each landmark separately
						smoothed_landmarks = []
						if right_hand_history:
							for i in range(len(pontos_mao)):
								# Collect all coordinates of a specific landmark
								x_values = [h[i]['x'] for h in right_hand_history]
								y_values = [h[i]['y'] for h in right_hand_history]
								z_values = [h[i]['z'] for h in right_hand_history]

								smoothed_landmarks.append({
									"index": pontos_mao[i],
									"x": np.mean(x_values),
									"y": np.mean(y_values),
									"z": np.mean(z_values)
								})
						
						last_known_right_hand = smoothed_landmarks
						hands_data[hand_label] = {"landmarks": smoothed_landmarks, "visible": True}

						right_hand_path_x.append(smoothed_landmarks[0]['x'])
						right_hand_path_y.append(smoothed_landmarks[0]['y'])
					
					elif hand_label == "Left":
						left_hand_history.append(landmarks)
						if len(left_hand_history) > FILTER_SIZE:
							left_hand_history.pop(0)
						
						smoothed_landmarks = []
						if left_hand_history:
							for i in range(len(pontos_mao)):
								x_values = [h[i]['x'] for h in left_hand_history]
								y_values = [h[i]['y'] for h in left_hand_history]
								z_values = [h[i]['z'] for h in left_hand_history]

								smoothed_landmarks.append({
									"index": pontos_mao[i],
									"x": np.mean(x_values),
									"y": np.mean(y_values),
									"z": np.mean(z_values)
								})

						last_known_left_hand = smoothed_landmarks
						hands_data[hand_label] = {"landmarks": smoothed_landmarks, "visible": True}
						
						left_hand_path_x.append(smoothed_landmarks[0]['x'])
						left_hand_path_y.append(smoothed_landmarks[0]['y'])

			# If a hand is not detected, send the last known position with visible: False
			if not current_detected_hands["Right"] and last_known_right_hand:
				hands_data["Right"] = {"landmarks": last_known_right_hand, "visible": False}
			
			if not current_detected_hands["Left"] and last_known_left_hand:
				hands_data["Left"] = {"landmarks": last_known_left_hand, "visible": False}

			# Send dictionary to Godot
			message = json.dumps(hands_data)
			client_socket.sendall((message + "\n").encode('utf-8'))

except KeyboardInterrupt:
	print("\nScript interrupted by user (Ctrl+C).")
except Exception as e:
	# An error occurred in the main loop or connection
	print(f"An error occurred: {e}")
finally:
	# Release resources and generate the graph
	if 'cap' in locals() and cap.isOpened():
		cap.release()
	client_socket.close()
	print("Resources released. Connection closed.")
	
#############################################################################
#                               Graph Generation                            #
#############################################################################
	
	if (right_hand_path_x or left_hand_path_x) and save_path:
		try:
			plt.figure(figsize=(10, 8))
			
			plt.gca().invert_yaxis()
			
			if right_hand_path_x:
				plt.plot(right_hand_path_x, right_hand_path_y, label='Right Hand', color='blue')
			
			if left_hand_path_x:
				plt.plot(left_hand_path_x, left_hand_path_y, label='Left Hand', color='red')
				
			# Graph title
			title = f"Hand Path | Patient: {patient_name}\nDate: {date_str} Time: {time_str}"
			plt.title(title)
			plt.xlabel('X Position')
			plt.ylabel('Y Position')
			plt.legend()
			plt.grid(True)
			
			plt.savefig(save_path) 
			print(f"Graph successfully saved at: {save_path}")
			plt.show()

		except Exception as e:
			print(f"ERROR while generating/saving the graph: {e}")
	elif not save_path:
		print("WARNING: Could not define the save path. The graph will not be generated/saved.")
