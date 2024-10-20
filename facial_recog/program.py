import face_recognition
import cv2
import numpy as np
import time

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it
known_image = face_recognition.load_image_file("test_2.jpg")  # Replace with your image file path
known_face_encoding = face_recognition.face_encodings(known_image)[0]  # Encode the known face

# Create an array of known face encodings and their names
known_face_encodings = [known_face_encoding]
known_face_names = ["Sean Regindin"]  # Replace with the actual name or label for the person

# Frame counter to track how many frames have passed
frame_counter = 0
process_every_n_frames = 30

# Variable to store the last recognized face name and a timer for how long to display it
last_recognized_name = None
last_seen_time = None
display_duration = 3  # Display name for 3 seconds after face disappears

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Increment frame counter
    frame_counter += 1

    # Only process every 30th frame
    if frame_counter % process_every_n_frames == 0:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Assume no faces are matched in this frame
        current_frame_recognized_name = None

        # Loop over the face locations and encodings
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the face encoding to the known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match was found in known_face_encodings, use the first one
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                # Store the recognized name and the time it was seen
                current_frame_recognized_name = name
                last_seen_time = time.time()  # Record the current time

            # Scale back up face locations since the frame was resized to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Update the last recognized name if a face was recognized in the current frame
        if current_frame_recognized_name:
            last_recognized_name = current_frame_recognized_name

    # Display the last recognized name if it exists and within the display duration
    if last_recognized_name and (time.time() - last_seen_time) < display_duration:
        # Draw a label with the name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, last_recognized_name, (50, 50), font, 1.0, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
