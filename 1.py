
import cv2
import mediapipe as mp

from playsound import playsound


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
flag1=0
flag=0
x2=0
y2=0
z2=0
"""t1 = gtts.gTTS("Correct")
# save the audio file
t1.save("correct.mp3")
t2=gtts.gTTS("Narrow")
t2.save("narrow.mp3")
t3=gtts.gTTS("Wider")
t3.save("wider.mp3")"""
count=0
# For static images:
"""IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
"""
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    x = str(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x)
    y = str(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y)
    z = str(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z)
    if flag==0: #Фикс плеча
        x3 = str(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
        y3 = str(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        z3 = str(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z)
        x1 = str(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x)
        y1 = str(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y)
        z1 = str(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z)
        flag=1
        flag1=1
        flag6=1
        flag5=1
    if flag1==1 and x[:3:]==x3[:3:] and y[:3:]==y3[:3:] :
        print("Правильно")
        flag1=2
        flag5=1
        if flag6==1:
            playsound('correct.mp3')
            flag6=2
    if flag1 == 1 and x[:3:] > x3[:3:]:
        print("Поставьте руки уже")
        flag6=1
        if flag5==1:
            flag5=2
            playsound('narrow.mp3')
    if flag1 == 1 and x[:3:] < x3[:3:]:
        print("Руки шире")
        flag6=1
        if flag5==1:
            flag5=2
            playsound('wider.mp3')
    if flag1 == 2 and x[:3:] == x3[:3:] and y[:3:] == y3[:3:] : #Верхняя точка
        flag1=3
    if flag1==3 and x[:3:]==x1[:3:] and y[:3:]==y1[:3:] : #нижняя точка 2
        flag1=1
        count+=1
        print(count)
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

