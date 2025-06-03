import cv2
import imutils
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from tkinter import messagebox, Tk
import os

# --- Email Configuration ---
EMAIL_SENDER = 'janasruthi41@gmail.com'
EMAIL_PASSWORD = 'gzod hmdf jmip qabd'  # App Password
EMAIL_RECEIVER = '@gmail.com'

def send_email_with_attachment(timestamp, image_path):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = "Motion Detected - Alert"
    body = f"Motion detected at {timestamp}. See attached image."
    msg.attach(MIMEText(body, 'plain'))

    with open(image_path, 'rb') as f:
        mime = MIMEBase('image', 'jpg', filename=os.path.basename(image_path))
        mime.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
        mime.add_header('X-Attachment-Id', '0')
        mime.add_header('Content-ID', '<0>')
        mime.set_payload(f.read())
        encoders.encode_base64(mime)
        msg.attach(mime)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("✅ Email with image sent.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def show_popup(message):
    root = Tk()
    root.withdraw()
    messagebox.showinfo("ALERT", message)
    root.destroy()

def log_event(timestamp):
    with open("log.txt", "a") as log_file:
        log_file.write(f"[{timestamp}] Motion Detected\n")

# --- Motion Detection Setup ---
cam = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
area = 1500
email_sent_once = False  # Ensure only first detection sends email
motion_counter = 0
motion_threshold = 5  # Require motion for 5 frames

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    text = "Normal"
    img = imutils.resize(img, width=1000)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    fgmask = fgbg.apply(gaussianImg)
    _, threshImg = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    threshImg = cv2.erode(threshImg, None, iterations=2)
    threshImg = cv2.dilate(threshImg, None, iterations=4)

    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"

    if text == "Moving Object detected":
        motion_counter += 1
    else:
        motion_counter = 0

    if motion_counter >= motion_threshold and not email_sent_once:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, img)

        send_email_with_attachment(timestamp, filename)
        show_popup(f"Motion Detected at {timestamp}")
        log_event(timestamp)
        email_sent_once = True

    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed", img)

    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
