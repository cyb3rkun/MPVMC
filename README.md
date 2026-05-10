# Features:
- **51 ARKit Blendshapes:** Perfect Sync facial tracking with a webcam
- **Head Transform:** Translation and Rotation with Quaternions.
- **Works on Linux**

# Setup

1. **Clone the repo:**
    ```bash
    git clone https://cyb3rkun/MPVMC.git
    ```
  
2. **Setup Virtual Environment:**
    ```bash
    cd MPVMC
    python3 -m venv MPVMCenv
    source ./MPVMCenv/bin/activate
    pip install mediapipe numpy python-osc python opencv-python
    ```
3. **Run the script:**
    ```bash
    python facetracker.py
    ```
4. **configure you're software:**
    - **IP:** 127.0.0.1
    - **PORT:** 39539

5. **Face Landmarker Task:**
	- This repo contains the mediapipe task file v2 that supports blendshapes and head transform matrixes. no need to download it manually,
	- If you want to download it manually, you can do so whith this command:
  		```fish
    	cd MPVMC && \
    	wget -O ./assets/face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  		```
   
7. **Enjoy you're vtubing journey with mediapipe on linux :)**
