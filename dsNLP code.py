import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


path="C:\\Users\\themj\\OneDrive\\Desktop\\SUMMER TRAINING\\Haar cascade Classifier\\haarcascade_frontalface_default.xml"


# algo loading:
classifier=cv.CascadeClassifier(path)


# def function to save images:
def user_guide():
    print("To capture image, press: c")
    print("To exit, press: x")
    print("."*30)

def save_image(frame,folder,image_name):
    if not os.path.exists(folder): # checking folder presence
        os.makedirs(folder)  # Create folder if it doesn't exist

    folder_length=len(os.listdir(folder)) +1 # +1 for image naming
    image_path=folder+"/"+image_name+str(folder_length)+".png"

    cv.imwrite(image_path,frame)

def take_selfie():
    user_guide()
    cam = cv.VideoCapture(0)
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)
            cv.imshow("Camera", image)
            key = cv.waitKey(20)

            if key == ord('x'):  # Terminate if input: x
                break
            
            if key == ord('c'):  # Take selfie and store it
                save_image(image, "MyPictures", "Selfie_")
                print("Selfie Taken")
    finally:
        # Always release the camera and close windows
        cam.release()
        cv.destroyAllWindows()


def color_filter(color):
    # cool(blue) tone filter:
    filter_frame=[]
    for i in range(480): # row
        temp=[]
        for j in range(640): # col
            temp.append(color)
        filter_frame.append(temp)

    filter_frame=np.array(filter_frame).astype(np.uint8)
    return filter_frame


def theme_filter():
    theme_path="theme1.jpg"
    
    theme_frame=cv.imread(theme_path)
    if len(theme_frame.shape) == 2:  # If theme is grayscale, convert it to BGR
        theme = cv.cvtColor(theme, cv.COLOR_GRAY2BGR)
    
    theme_frame=cv.resize(theme_frame,(640,480))
    return theme_frame

def filter_selfie(color, filter_type):
    cam = cv.VideoCapture(0)
    if (filter_type=="color"):
        filter_frame=color_filter(color)
    elif (filter_type=="theme"):
        filter_frame=theme_filter()
    else:
        print("Invalid filter type!")
        return
    
    try:
        user_guide()
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            image=cv.flip(img,1)
            filter_image=cv.addWeighted(image,0.8,filter_frame,0.3,1) #(real-img(alpha), ri-portion, filter-img(gamma), f-portion, adjust-alpha)
    
            cv.imshow("Filter",filter_image)
    
            key = cv.waitKey(20)

            if key == ord('x'):  # Terminate if input: x
                break
            
            if key == ord('c'):  # Take selfie and store it
                save_image(filter_image, "MyPictures", "Filter_")
                print("Filtered Selfie Taken")
    finally:
        # Always release the camera and close windows
        cam.release()
        cv.destroyAllWindows()


def face_detection():
    cam = cv.VideoCapture(0)
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)
            
            faces=classifier.detectMultiScale(img,1.1,5)

            try: #(if any error or face not detected)
                for (x,y,w,h) in faces:
                    cv.rectangle(image,(x,y),(x+w,y+h),(200,100,50),4)
            except:
                pass
    
            cv.imshow("Face detection",image)
            key = cv.waitKey(20)

            if key == ord('x'):  # Terminate if input: x
                break
            
            if key == ord('c'):  # Take selfie and store it
                save_image(image, "MyPictures", "FaceDetection_")
                print("Face detected image saved")
    finally:
        # Always release the camera and close windows
        cam.release()
        cv.destroyAllWindows()


        
def edge_detection():
    cam = cv.VideoCapture(0)
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)
            
            edge_image=cv.Canny(image,100,200)
    
            cv.imshow("Edge detection",edge_image)
            key = cv.waitKey(20)

            if key == ord('x'):  # Terminate if input: x
                break
            
            if key == ord('c'):  # Take selfie and store it
                save_image(edge_image, "MyPictures", "EdgeDetection_")
                print("Edge detected image saved")
    finally:
        # Always release the camera and close windows
        cam.release()
        cv.destroyAllWindows()
        

def brightness_control(adjust):
    cam = cv.VideoCapture(0)
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)

            if (adjust=="lower"):
                adjusted_image=image.copy()*0.7
                adjusted_image[adjusted_image<0]=0 
                adjusted_image=adjusted_image.astype(np.uint8)
                pass
            elif (adjust=="higher"):
                adjusted_image=image.copy()*1.5
                adjusted_image[adjusted_image>255]=255 
                adjusted_image=adjusted_image.astype(np.uint8)
            else:
                print("Invalid adjustment type!")
                return
            
    
            cv.imshow("Original",image)
            cv.imshow("Brightness adjusted",adjusted_image)
            
            key = cv.waitKey(20)

            if key == ord('x'):  # Terminate if input: x
                break
            
            if key == ord('c'):  # Take selfie and store it
                save_image(adjusted_image, "MyPictures", "BrightnessAdjusted_")
                print("Brightness adjusted image saved")
    finally:
        # Always release the camera and close windows
        cam.release()
        cv.destroyAllWindows()



def face_blur():
    cam = cv.VideoCapture(0)
    while True:
        __, img = cam.read()

        image = cv.flip(img, 1)
        faces = classifier.detectMultiScale(img, 1.1, 5)
        
        # Handle potential errors or face detection issues
        faceCrop = None
        try:
            for face in faces:
                if face[-1] == max(faces[:, -1]):
                    faceCrop = face
                    break
            
            if faceCrop is not None:
                x = faceCrop[0]
                y = faceCrop[1]
                w = faceCrop[2]
                h = faceCrop[3]
                faceCrop = image[y-10:y+h, x:x+w, :]

                # Blurring cropped face
                blur_image = cv.blur(faceCrop, (16, 16))

                # Blurring face on original image
                image[y-10:y+h, x:x+w, :] = blur_image

        except:
            pass
        
        cv.imshow("Blur face", image)
        
        key = cv.waitKey(30)

        if key == ord('x'):  # Terminate if input: x
            break
        
        if key == ord('c'):  # Take selfie and store it
            save_image(image, "MyPictures", "BlurFace_")
            print("Blur face image saved")
    
    # Always release the camera and close windows
    cam.release()
    cv.destroyAllWindows()
        

def masked_image():
    cam = cv.VideoCapture(0)
    lower=np.array([180,200,200])
    upper=np.array([255,255,255])
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)

            mask_img=cv.inRange(image,lower,upper) # masking for color range to black
            cv.imshow('Masked image',mask_img)
            
            key = cv.waitKey(20)

            if key == ord('x'):  # Terminate if input: x
                break
            
            if key == ord('c'):  # Take selfie and store it
                save_image(mask_img, "MyPictures", "MaskedImage_")
                print("Masked image saved")
    finally:
        # Always release the camera and close windows
        cam.release()
        cv.destroyAllWindows()



def black_white():
    cam = cv.VideoCapture(0)
    lower=np.array([180,200,200])
    upper=np.array([255,255,255])
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(img, 1)

            gray_image=cv.cvtColor(image,cv.COLOR_RGB2GRAY)
            cv.imshow('Black and white',gray_image)
            
            key = cv.waitKey(20)

            if key == ord('x'):  # Terminate if input: x
                break
            
            if key == ord('c'):  # Take selfie and store it
                save_image(gray_image, "MyPictures", "BlackAndWhite_")
                print("Black and white image saved")
    finally:
        # Always release the camera and close windows
        cam.release()
        cv.destroyAllWindows()


def rgb_channels():
    # Open the camera
    cam = cv.VideoCapture(0)

    try:
        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            img = cv.flip(img, 1)

            # Extract R, G, B channels by zeroing out the other channels
            r = img.copy()
            g = img.copy()
            b = img.copy()

            r[:, :, 1] = 0  # Zero out the green channel
            r[:, :, 2] = 0  # Zero out the blue channel

            g[:, :, 0] = 0  # Zero out the blue channel
            g[:, :, 2] = 0  # Zero out the red channel

            b[:, :, 0] = 0  # Zero out the green channel
            b[:, :, 1] = 0  # Zero out the red channel

            # Display the original and color-separated frames
            cv.imshow("Original", img)
            cv.imshow("Red Channel", r)
            cv.imshow("Green Channel", g)
            cv.imshow("Blue Channel", b)

            key = cv.waitKey(20)
            if key == ord('x'):  # Terminate if input: 'x'
                break

    finally:
        # Release the camera and close all OpenCV windows
        cam.release()
        cv.destroyAllWindows()


print("#"*50)
print()
print(" "*12,"... Camera program ...")

run=1
choice=0

while(run==1):
    print("#"*50)
    print()
    print(">>> Options:\n")
    print("1: Take selfie")
    print("2: Filter")
    print("3: Face detection")
    print("4: Edge detection")
    print("5: Adjust brightness")
    print("6: Face blurring")
    print("7: Masked image")
    print("8: Monochrome (Black & White)")
    print("9: Color extraction from image")
    print("0: Exit\n")

    choice=input("Enter choice: ")
    print("*"*50)
    if choice == '0':
        run=0
        print(" "*15,"... Exit ...")
        print("#"*50)
        break
    
    if choice == '1':
        user_guide()
        take_selfie()
        
    elif choice == '2':
        print(">>> Filter types:")
        print("1. cool(blue) filter")
        print("2. warm(yellow) filter")
        print("3. mixed color filter")
        print()
        
        filter_choice=input("Choose filter type: ")
        if filter_choice == '1':
            print("You selected: cool filter")
            print("."*30)
            user_guide()
            blue=[252,215,139] # light blue color
            filter_selfie(blue,"color")
        elif filter_choice == '2':
            print("You selected: warm filter")
            print("."*30)
            user_guide()
            yellow=[139,206,247] # light yellow color
            filter_selfie(yellow,"color")
        elif filter_choice == '3':
            print("You selected: theme filter")
            print("."*30)
            user_guide()
            filter_selfie("mixed","theme")
        else:
            print("Invalid choice!")

    elif choice == '3':
        user_guide()
        face_detection()
        
    elif choice == '4':
        user_guide()
        edge_detection()
        
    elif choice == '5':
        print(">>> Brightness adjustment:")
        print("1. lower brightness")
        print("2. increase brightness")
        print()
        
        filter_choice=input("Choose filter type: ")
        
        if filter_choice == '1':
            print("You selected: lower brightness")
            print("."*30)
            user_guide()
            brightness_control("lower")
        elif filter_choice == '2':
            print("You selected: increase brightness")
            print("."*30)
            user_guide()
            brightness_control("higher")
        else:
            print("Invalid choice!")
            
    elif choice == '6':
        face_blur()
        user_guide()
        
    elif choice == '7':
        masked_image()
        user_guide()
        
    elif choice == '8':
        black_white()
        user_guide()
        
    elif choice == '9':
        rgb_channels()
        print("To exit, press: x")
        print("."*30)
        
    else:
        print()
        print("!!! Invalid choice! Try Again...")
        print()

plt.show()
