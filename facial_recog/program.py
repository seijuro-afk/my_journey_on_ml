import face_recognition
import cv2

# Load a sample picture and create encodings for it
known_image = face_recognition.load_image_file("test_2.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Store known face encodings and names
known_face_encodings = [known_face_encoding]
known_face_names = ["Known Person"]

# Initialize variables for face recognition
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Access the webcam feed
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize the frame to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (OpenCV uses) to RGB color (face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Process every frame (skip frames for performance if necessary)
    if process_this_frame:
        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        
        # Ensure there are faces detected before proceeding
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        else:
            face_encodings = []
        
        face_names = []
        for face_encoding in face_encodings:
            # Compare the face encoding with the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)

            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
