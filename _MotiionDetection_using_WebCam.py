import cv2,time

def motionDetection():
    # Create a video capture object
    video = cv2.VideoCapture(0)

    first_frame = None

    while True:
         # check is a boolean value, frame is the image
        check, frame = video.read()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(frame,"Sai Nithin", (350, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 3)
        cv2.putText(frame,"S20200010067", (350, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if first_frame is None:
            first_frame = gray
            continue
        delta_frame = cv2.absdiff(first_frame,gray)
        
        # Apply a threshold to the foreground mask
        threshold_frame = cv2.threshold(delta_frame, 100, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate the thresholded image to fill in holes
        threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
        
        # Find contours in the thresholded image
        (cntr,_) = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw a rectangle around the contours
        for contour in cntr:
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
        
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        
        # Break the loop if the 'q' key is pressed
        if key == ord('q'):
            break
        
    video.release()
    # Close all windows
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    motionDetection()