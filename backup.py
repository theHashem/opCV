import cv2
import numpy as np

def bild():
    img = cv2.imread("yellow1.jpg")

    lower_blue = np.array([0, 84, 159])
    upper_blue = np.array([142, 186, 299])
    blue_mask = cv2.inRange(img, lower_blue, upper_blue)

    lower_yellow = np.uint8([190, 190, 0])
    upper_yellow = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)

    lower_white = np.uint8([200, 200,   0])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(img, lower_white, upper_white)


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #edges = cv2.Canny(gray, 75, 150)
    #mask = cv2.inRange(img, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, blue_mask)
    masked = cv2.bitwise_and(img, img, mask=mask)
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blur, 200, 180)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=200)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    
    cv2.imshow("hls", hls)
    #cv2.imshow("masked", masked)
    #cv2.imshow("Edges", edges)
    cv2.imshow("Image", img)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def select_rgb_white_yellow(): 
    image = cv2.imread("street.png")
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    
    cv2.imshow("masked", masked)

def gpt():
    img = cv2.imread('street.png')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def senkrecht():
    img = cv2.imread('yellow1.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    blur = cv2.GaussianBlur(mask, (5, 5), 0)

    edges = cv2.Canny(blur, 500, 300)

    #lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, maxLineGap=500)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < abs(y1 - y2) and abs(y1 - y2) > 30:
            filtered_lines.append(line)


    if lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            #cv2.line(img, (x1, y1), (x2-50, y2),  (255, 255, 0), 3)
            print(x1,y1, x2, y2)

    # Anzeigen des Ergebnisbildes
    cv2.imshow("Blur", blur)
    cv2.imshow("Edges", edges)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def line_vedio_detection():
    lower_yellow = np.array([10, 100, 20])
    upper_yellow = np.array([40, 255, 255])

    video = cv2.VideoCapture(0)

    while True:
        success, img = video.read()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blur, 500, 300)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=30, maxLineGap=10)

        if lines is not None:

            filtered_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < abs(y1 - y2) and abs(y1 - y2) > 30:
                    filtered_lines.append(line)

            for line in filtered_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 8)

                print(x1, y1, x2, y2)
                

        #print(lines[0][0])
        #img = cv2.line(img,(lines[0][0][0], lines[0][0][1] ),(lines[0][0][2], lines[0][0][3]),(255,0,0),5)
  
        cv2.imshow("edges", edges)
        cv2.imshow("mask", mask)
        cv2.imshow("webcam", img)
        cv2.waitKey(1)

if __name__ == '__main__':

    #select_rgb_white_yellow()
    #bild()
    #gpt()
    #senkrecht()
    line_vedio_detection()