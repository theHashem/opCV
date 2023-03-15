import cv2
import numpy as np

def line_vedio_detection():
    lower = np.array([0, 84, 159]) #blaue Farbe
    upper = np.array([142, 186, 299])

    video = cv2.VideoCapture(0)

    x1_distance=0
    x2_distance=0
    detected=False
    distance=0

    while True:
        success, img = video.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image, lower, upper)

        edges = cv2.Canny(mask, 75, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=80)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 8)
            
        """if lines is not None:
            x1 = lines[0][0][0]
            y1 = lines[0][0][1]    
            x2 = lines[0][0][2]
            y2 = lines[0][0][3]

            if detected == False:

                if x1_distance == 0:
                    x1_distance=x1
                    x2_distance=x1

                if x1 != x2_distance:
                    x2_distance=x1
                    distance= abs(x1_distance - x2_distance)
                    detected=True"""

        print(x1)

        #print(lines[0][0])
        #img = cv2.line(img,(lines[0][0][0], lines[0][0][1] ),(lines[0][0][2], lines[0][0][3]),(255,0,0),5)
  
        #cv2.imshow("edges", edges)
        cv2.imshow("mask", mask)
        cv2.imshow("webcam", img)
        cv2.waitKey(1)

def line_on_pic():

    img = cv2.imread('street_lines.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=80)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        img = cv2.line(img,(x1, y1),(x2, y2),(255,0,0),5)

    
    cv2.imshow("Edges", edges)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def line2():

    img = cv2.imread('street_lines.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=70)


    print(lines[0][0])
    cv2.line(img,(lines[0][0][0], lines[0][0][1] ),(lines[0][0][2], lines[0][0][3]),(255,0,0),5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    if len(lines) == 2:
        x1, y1, x2, y2 = lines[0][0]
        x3, y3, x4, y4 = lines[1][0]
        distance = abs((y2-y1) - (y4-y3))
        print(f'Der Abstand ist {distance:.2f}')
    else:
        print('Es gibt mehr als zwei oder keine Linie.')
    
    #print(lines)

    cv2.imshow('Ergebnis', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':

    line_vedio_detection()
    #line_on_pic()
    #line2()
