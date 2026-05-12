import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision


MODEL_PATH = "./assets/face_landmarker_v2_with_blendshapes.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# rendering faceMesh
VMC_IP = "127.0.0.1"
VMC_PORT = 39539

detection_result = None

from pythonosc import osc_bundle_builder, osc_message_builder, udp_client

client = udp_client.SimpleUDPClient(VMC_IP, VMC_PORT)


def extract_vmc_transform(matrix):
	"""
	Extracts Translation and Quaternion Rotation from a 4x4 matrix.
	MediaPipe Matrix format:
	[ R00  R01  R02  Tx ]
	[ R10  R11  R12  Ty ]
	[ R20  R21  R22  Tz ]
	[  0    0    0   1  ]
	"""
	# 1. Extract Translation
	# We flip Z because MediaPipe's 'forward' is negative Z
	tx = matrix[0, 3]
	ty = matrix[1, 3]
	tz = -matrix[2, 3]

	# 2. Extract Rotation Matrix (Top-left 3x3)
	m = matrix[0:3, 0:3]

	# 3. Convert 3x3 Rotation Matrix to Quaternion (qx, qy, qz, qw)
	tr = m[0, 0] + m[1, 1] + m[2, 2]
	if tr > 0:
		s = np.sqrt(tr + 1.0) * 2
		qw = 0.25 * s
		qx = (m[2, 1] - m[1, 2]) / s
		qy = (m[0, 2] - m[2, 0]) / s
		qz = (m[1, 0] - m[0, 1]) / s
	elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
		s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
		qw = (m[2, 1] - m[1, 2]) / s
		qx = 0.25 * s
		qy = (m[0, 1] + m[1, 0]) / s
		qz = (m[0, 2] + m[2, 0]) / s
	elif m[1, 1] > m[2, 2]:
		s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
		qw = (m[0, 2] - m[2, 0]) / s
		qx = (m[0, 1] + m[1, 0]) / s
		qy = 0.25 * s
		qz = (m[1, 2] + m[2, 1]) / s
	else:
		s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
		qw = (m[1, 0] - m[0, 1]) / s
		qx = (m[0, 2] + m[2, 0]) / s
		qy = (m[1, 2] + m[2, 1]) / s
		qz = 0.25 * s

    # NOTE: uncomment or comment these if the head of you're
    # characters's x rotation (yaw) is flipped from you're eyes.
	qy = -qy
	qz = -qz
	return tx, ty, tz, qx, qy, qz, qw


def send_vmc(result):
	if not result.face_blendshapes:
		return
	bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
	for shape in result.face_blendshapes[0]:
		msg = osc_message_builder.OscMessageBuilder(address="/VMC/Ext/Blend/Val")
		msg.add_arg(shape.category_name)
		msg.add_arg(float(shape.score))
		bundle.add_content(msg.build())

	if result.facial_transformation_matrixes:
		matrix = result.facial_transformation_matrixes[0]
		tx, ty, tz, qx, qy, qz, qw = extract_vmc_transform(matrix)
		msg = osc_message_builder.OscMessageBuilder(address="/VMC/Ext/Bone/Pos")
		msg.add_arg("Head")
		msg.add_arg(float(tx))
		msg.add_arg(float(ty))
		msg.add_arg(float(tz))

		msg.add_arg(float(qx))
		msg.add_arg(float(qy))
		msg.add_arg(float(qz))
		msg.add_arg(float(qw))
		bundle.add_content(msg.build())

	msg = osc_message_builder.OscMessageBuilder(address="/VMC/Ext/OK")
	msg.add_arg(1)
	bundle.add_content(msg.build())
	client.send(bundle.build())


# Create a face landmarker instance with the live stream mode:
def res_callback(
	result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
	# print("face landmarker result: {}".format(result))
	global detection_result
	detection_result = result
	send_vmc(result)
	# TODO:
	# send results over VMC


options = FaceLandmarkerOptions(
	base_options=BaseOptions(model_asset_path=MODEL_PATH),
	running_mode=VisionRunningMode.LIVE_STREAM,
	output_face_blendshapes=True,
	output_facial_transformation_matrixes=True,
	result_callback=res_callback,
)


ESC = 27

from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

import numpy as np

import mediapipe as mp
import matplotlib.pyplot as plt


def draw_landmarks_on_image(rgb_image, detection_result):
	face_landmarks_list = detection_result.face_landmarks
	annotated_image = np.copy(rgb_image)

	# Loop through the detected faces to visualize.
	for idx in range(len(face_landmarks_list)):
		face_landmarks = face_landmarks_list[idx]

		# Draw the face landmarks.

		drawing_utils.draw_landmarks(
			image=annotated_image,
			landmark_list=face_landmarks,
			connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
			landmark_drawing_spec=None,
			connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
		)
		drawing_utils.draw_landmarks(
			image=annotated_image,
			landmark_list=face_landmarks,
			connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
			landmark_drawing_spec=None,
			connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
		)
		drawing_utils.draw_landmarks(
			image=annotated_image,
			landmark_list=face_landmarks,
			connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
			landmark_drawing_spec=None,
			connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
		)
		drawing_utils.draw_landmarks(
			image=annotated_image,
			landmark_list=face_landmarks,
			connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
			landmark_drawing_spec=None,
			connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
		)

	return annotated_image


def draw_blendshapes_on_frame(frame, blendshapes):
	"""Draws a compact two-column live bar graph directly on the frame."""
	h, w, _ = frame.shape

	# Layout Settings
	num_blendshapes = len(blendshapes)
	col1_limit = (num_blendshapes + 1) // 2  # Split roughly in half

	bar_height = 12
	bar_width_max = 120
	padding = 4

	font_scale = 0.3
	font_thickness = 1
	text_color = (255, 255, 255)
	bar_bg_color = (40, 40, 40)
	bar_fill_color = (
		180,
		180,
		0,
	)  # Warudo-ish Blue/Orange? Let's go with a nice Teal (180, 180, 0)

	for i, category in enumerate(blendshapes):
		name = category.category_name
		score = category.score

		# Determine Column
		col = 0 if i < col1_limit else 1
		row = i if col == 0 else i - col1_limit

		# Calculate Coordinates
		x_start = 10 + (col * (bar_width_max + 10))
		y_start = 30 + (row * (bar_height + padding))

		# 1. Draw Background (Full Bar)
		cv2.rectangle(
			frame,
			(x_start, y_start),
			(x_start + bar_width_max, y_start + bar_height),
			bar_bg_color,
			-1,
		)

		# 2. Draw Value (Filled Bar)
		current_width = int(bar_width_max * score)
		if current_width > 0:
			cv2.rectangle(
				frame,
				(x_start, y_start),
				(x_start + current_width, y_start + bar_height),
				bar_fill_color,
				-1,
			)

		# 3. Draw Text inside or next to the bar
		# We put the text slightly offset from the start of the bar
		display_text = f"{name}: {score:.2f}"
		cv2.putText(
			frame,
			display_text,
			(x_start + 5, y_start + 8),
			cv2.FONT_HERSHEY_SIMPLEX,
			font_scale,
			text_color,
			font_thickness,
		)

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
# cap.set(cv2.CAP_PROP_FPS, 30)

with FaceLandmarker.create_from_options(options) as landmarker:
	while cap.isOpened():
		success, frame = cap.read()
		# skip empty frames
		if not success:
			print("Ignoring empty camera frame.")
			continue

			# prepare image to run inference
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
		frame_timestamp = int(time.time() * 1000)

		# run inference
		landmarker.detect_async(mp_image, frame_timestamp)

		# # NOTE: comment out this block to disable annotations
		# if detection_result and detection_result.face_landmarks:
		# 	annotated_frame_rgb = draw_landmarks_on_image(
		# 		mp_image.numpy_view(), detection_result
		# 	)
		# 	frame = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)
		# 	draw_blendshapes_on_frame(frame, detection_result.face_blendshapes[0])
		#
		# # Render Result onto webcam frame.
		#
		# # NOTE: comment out this line to not show image
		# cv2.imshow("WarudoMP", frame)

		if cv2.waitKey(1) & 0xFF == ESC:
			break

cap.release()
