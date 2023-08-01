import random
import pygame
import cv2
import numpy as np
import time

# Initialize
pygame.init()

# Create Window/Display
width, height = 1280, 720
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Balloon Pop")

# Initialize Clock for FPS
fps = 30
clock = pygame.time.Clock()

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

# Images
imgBalloon = pygame.image.load("./Resources/BalloonRed.png").convert_alpha()
rectBalloon = imgBalloon.get_rect()
rectBalloon.x, rectBalloon.y = 500, 100

# Variables
speed = 15
score = 0
startTime = time.time()
totalTime = 60

# Detector


points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

# Calibration
cv2.namedWindow("Webcam Calibration")
cv2.setMouseCallback("Webcam Calibration", mouse_callback)

while len(points) < 4:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow("Webcam Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

# Calibration transformation matrix
src_points = np.float32(points)
dst_points = np.float32([(0, 0), (width, 0), (width, height), (0, height)])
M = cv2.getPerspectiveTransform(src_points, dst_points)

def resetBalloon():
    rectBalloon.x = random.randint(100, width - 100)
    rectBalloon.y = height + 50

def balloonBurst():
    resetBalloon()

# Main loop
start = True
while start:
    # Get Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            start = False
            pygame.quit()

    # Apply Logic
    timeRemain = int(totalTime - (time.time() - startTime))
    if timeRemain < 0:
        window.fill((255, 255, 255))

        font = pygame.font.Font(None, 50)
        textScore = font.render(f"Your Score: {score}", True, (50, 50, 255))
        textTime = font.render(f"Time UP", True, (50, 50, 255))
        window.blit(textScore, (450, 350))
        window.blit(textTime, (530, 275))

    else:
        # OpenCV
        success, img = cap.read()
        img = cv2.flip(img, 1)


        # Yellow ball detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Assuming the largest contour is the yellow ball
            max_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(max_contour)
            x, y, radius = int(x), int(y), int(radius)

            # Draw yellow ball on the webcam feed
            if radius > 5:
                cv2.circle(img, (x, y), radius, (0, 255, 255), 2)
                if rectBalloon.collidepoint(x, y):
                    balloonBurst()
                    score += 10
                    speed += 1

        rectBalloon.y -= speed  # Move the balloon up
        # check if balloon has reached the top without pop
        if rectBalloon.y < 0:
            resetBalloon()
            speed += 1

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        calibrated_frame = cv2.warpPerspective(imgRGB, M, (width, height))

        # Rotate the calibrated_frame using cv2.rotate
        calibrated_frame = cv2.rotate(calibrated_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame = pygame.surfarray.make_surface(calibrated_frame).convert()
        frame = pygame.transform.flip(frame, True, False)
        window.blit(frame, (0, 0))
        window.blit(imgBalloon, rectBalloon)

        font = pygame.font.Font(None, 50)
        textScore = font.render(f"Score: {score}", True, (50, 50, 255))
        textTime = font.render(f"Time: {timeRemain}", True, (50, 50, 255))
        window.blit(textScore, (35, 35))
        window.blit(textTime, (1000, 35))

    # Update Display
    pygame.display.update()
    # Set FPS
    clock.tick(fps)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
