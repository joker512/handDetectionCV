DEBUG=-g -DDEVEL
RELEASE=-O3

all: release

debug:
	g++ $(DEBUG) -std=c++11 -o shooting_hand main.cpp ShootingChecker.cpp ImageReader.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui
release:
	g++ $(RELEASE) -std=c++11 -o shooting_hand main.cpp ShootingChecker.cpp ImageReader.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui
