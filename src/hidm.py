# USAGE
# python hidm.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

from src.detect import Det


if __name__ == '__main__':
    paths = ["MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel"]

    det = Det(prototxt=paths[0], model=paths[1])

    video = "../data/crop1/crop.mp4"
    det.run(video=video)