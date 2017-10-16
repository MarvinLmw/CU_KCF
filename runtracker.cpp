#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"
/*
#include <dirent.h>

#include <sys/time.h>*/

using namespace std;
using namespace cv;

int part_time[16] = {0, 0, 0, 0,
			0,0,0,0,
			0,0,0,0,
			0,0,0,0};

KCFTracker kcf_init(){
	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	return tracker;
}


int main(int argc, char* argv[]){
  //struct timeval tv, tz,tv0, tz0;

	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}

	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;

	// Path to list.txt
	ifstream listFile;
	string fileName = "D:/Project_Lmw/KCFcpp-master/src/images.txt";
  	listFile.open(fileName);

  	// Read groundtruth for the 1st frame
  	ifstream groundtruthFile;
	string groundtruth = "D:/Project_Lmw/KCFcpp-master/src/region.txt";
  	groundtruthFile.open(groundtruth);
  	string firstLine;
  	getline(groundtruthFile, firstLine);
	groundtruthFile.close();

  	istringstream ss(firstLine);

  	// Read groundtruth like a dumb
  	float x1, y1, x2, y2, x3, y3, x4, y4;
  	char ch;
	ss >> x1;
	ss >> ch;
	ss >> y1;
	ss >> ch;
	ss >> x2;
	ss >> ch;
	ss >> y2;
	ss >> ch;
	ss >> x3;
	ss >> ch;
	ss >> y3;
	ss >> ch;
	ss >> x4;
	ss >> ch;
	ss >> y4;


	// Using min and max of X and Y for groundtruth rectangle
	float xMin =  min(x1, min(x2, min(x3, x4)));
	float yMin =  min(y1, min(y2, min(y3, y4)));
	float width = max(x1, max(x2, max(x3, x4))) - xMin;
	float height = max(y1, max(y2, max(y3, y4))) - yMin;


	// Read Images
	ifstream listFramesFile;
	string listFrames = "D:/Project_Lmw/KCFcpp-master/src/images.txt";
	listFramesFile.open(listFrames);
	string frameName;


	// Write Results
	ofstream resultsFile;
	string resultsPath = "D:/Project_Lmw/KCFcpp-master/src/output.txt";
	resultsFile.open(resultsPath);


	// Frame counter
	int nFrames = 0;
	char name_write[15] = {};
	int ii=0;
        //gettimeofday(&tv0, NULL);
	while ( getline(listFramesFile, frameName) ){
		frameName = frameName;
		ii++;
		if(ii>200) break;
      		//gettimeofday(&tv, NULL);
		// Read each frame from the list
		frame = imread(frameName, 1);
		//Mat frame1 = imread("D:\\Project_Lmw\\1\\0001.jpg", 0);
		//CV_LOAD_IMAGE_COLOR
      		//gettimeofday(&tz, NULL);
      		//cout<<"imread = "<<-1*(tv.tv_sec*1000+tv.tv_usec/1000-tz.tv_sec*1000-tz.tv_usec/1000)<<endl;

		// First frame, give the groundtruth to the tracker
		if (nFrames == 0) {
			tracker.init( Rect(xMin, yMin, width, height), frame );
			rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
			resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
		}
		// Update
		else{
                        //gettimeofday(&tv, NULL);
	if(tracker.initial){
		tracker.Allocate();
		tracker.initial = false;
	}
			result = tracker.update(frame);
			//cout<<frame<<" = ";
      			//gettimeofday(&tz, NULL);
			if(tracker.nFrame_lose>600){
				cout<<"//************************lose the target!!*************************//"<<endl;
				break;
			}
      			//cout<<"update = "<<-1*(tv.tv_sec*1000+tv.tv_usec/1000-tz.tv_sec*1000-tz.tv_usec/1000)<<"ms ";
			if(!tracker.target_lose)
			  rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 4, 8 );
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
		}
		nFrames++;
		//if(nFrames == 150)
		//	break;
		if (!SILENT){
			namedWindow("Image", 0);
			imshow("Image", frame);
			waitKey(1);
			sprintf(name_write, "%04d.jpg", nFrames);
			//imwrite(name_write, frame);
		}
	}
      /*gettimeofday(&tz0, NULL);
      cout<<"SDL_UpdateTexture = "<<-1*(tv0.tv_sec*1000+tv0.tv_usec/1000-tz0.tv_sec*1000-tz0.tv_usec/1000)<<endl;
      cout<<"detect = "<<part_time[0]<<endl;
      cout<<"train_features = "<<part_time[4]<<endl;
      cout<<"train = "<<part_time[1]<<endl;
      cout<<"feature_init = "<<part_time[5]<<endl;
      cout<<"feature_init_subwindow = "<<part_time[7]<<endl;
      cout<<"copyMakeBorder = "<<part_time[8]<<endl;
      cout<<"feature_hog = "<<part_time[6]<<endl;
      cout<<"feature_total = "<<part_time[2]<<endl;


      cout<<"copyMakeBorder_8u = "<<part_time[9]<<endl;
      cout<<"copyMakeBorder_8u_1 = "<<part_time[10]<<endl;
      cout<<"copyMakeBorder_8u_2 = "<<part_time[11]<<endl;*/

	resultsFile.close();

	listFile.close();

}
