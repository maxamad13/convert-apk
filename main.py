import cv2
import mediapipe as mp
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

class PoseDetectorApp(App):
    def build(self):
        self.up = False
        self.counter = 0

        # Create the root layout
        self.layout = BoxLayout(orientation='vertical')

        # Create the image widget to display the camera feed
        self.image = Image()
        self.layout.add_widget(self.image)

        # Create a label to display the counter
        self.label = Label(text="Counter: 0", font_size=30)
        self.layout.add_widget(self.label)

        # Initialize the camera (0 is typically the default camera, adjust if needed)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open the camera.")
            exit()

        # Schedule the update method to run at a fixed interval
        Clock.schedule_interval(self.update, 1 / 30)  # Update at 30 FPS

        return self.layout

    def update(self, dt):
        success, img = self.cap.read()

        if not success:
            print("Error: Could not read a frame.")
            return

        img = cv2.resize(img, (1280, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            points = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                points[id] = (cx, cy)

            cv2.circle(img, points[12], 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, points[14], 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, points[11], 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, points[13], 15, (255, 0, 0), cv2.FILLED)

            if not self.up and points[14][1] + 40 < points[12][1]:
                print("UP")
                self.up = True
                self.counter += 1
            elif points[14][1] > points[12][1]:
                print("Down")
                self.up = False

        cv2.putText(img, str(self.counter), (100, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 12)

        # Display the updated image
        self.image.texture = self.texture_from_cv2image(img)

        # Update the label text
        self.label.text = f"Counter: {self.counter}"

    def texture_from_cv2image(self, img):
        buf = cv2.flip(img, 0).tostring()
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def on_stop(self):
        # Release the camera when the app is stopped
        self.cap.release()

if __name__ == '__main__':
    PoseDetectorApp().run()
