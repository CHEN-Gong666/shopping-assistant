import cv2
import tfPredict
from PIL import Image

def printResultOnScreen(frame, result):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                result,
                (50, 50),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)

def validateFrame(frame):
    image = Image.fromarray(frame, 'RGB')
    className = tfPredict.predict(image)
    printResultOnScreen(frame, className)

def openCamera():
    video_capture = cv2.VideoCapture(0)

    if video_capture.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
        while (True):
            ret, frame = video_capture.read()
            if ret:
                validateFrame(frame)
                cv2.imshow("Coffee mug on live!", frame)

                # save frame
                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error : Failed to capture frame")

        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("cannot open camera")


def main():
    openCamera()


if __name__ == "__main__":
    main()
