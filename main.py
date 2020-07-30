import face_recognition
import cv2
import numpy as np
import json
import time
    
video_capture = cv2.VideoCapture(0)


#sophie_image = face_recognition.load_image_file("Sophie.jpg")
#sophie_face_encoding = face_recognition.face_encodings(sophie_image)[0]

#lists = sophie_face_encoding.tolist()
#with open('sophie_face_encoding.json', 'w') as f:
    #json.dump(lists, f)

with open("sophie_face_encoding.json", "r") as read_file:
    sophie_face_encoding = json.load(read_file)
    
#Create arrays of known face encodings and their names
known_face_encodings = [
    sophie_face_encoding
]
known_face_names = [
    "Sophie"
]
#for new visitors
text_v = "visitor_"
i=1
#lists for tracking person
names_on_screen = []
prev_names_on_screen = []

while True:
    
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "new visitor"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        names_on_screen.append(name)

        if name=="new visitor":
            new_visitor_encoding = face_encoding
            
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 17), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 255, 0), 1)

    if ((len(prev_names_on_screen)-len(names_on_screen)>0)&("new visitor" in prev_names_on_screen)):
        print ('visitor is gone')
        
        #generate name
        text=text_v + str(i)
        i = i+1
        #print(text)
            
        #save to json face_encoding of this person
        lists = new_visitor_encoding.tolist()
        with open(text + '_encoding.json', 'w') as f:
            json.dump(lists, f)

        #add to lists
        known_face_names.append(text)
        known_face_encodings.append(new_visitor_encoding)
    
    prev_names_on_screen = names_on_screen
    names_on_screen = []
    
    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
