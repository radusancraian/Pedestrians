// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

double lab4PatternRecognition(Mat imagine1, Mat imagine2);
const int nrPictures = 34;
double distanceTable[34][34];

void createPedestriansTable(){
	char fname[256];

	Mat pictureVector[nrPictures];

	double distanceTable[128][128];
	FILE *f;
	f = fopen("result.txt", "w+");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	for (int i = 0; i <= 127; i++){
		sprintf(fname, "Images/templates/t%dzoom.bmp", i+1);
		pictureVector[i] = imread(fname, 0);
	}

	for (int i = 0; i < 127; i++){
		for (int j = i + 1; j < 128; j++){
			distanceTable[i][j] = float(lab4PatternRecognition(pictureVector[i], pictureVector[j]));
			distanceTable[j][i] = distanceTable[i][j];
			printf("%d %d %f\n", i, j, distanceTable[i][j]);
			fprintf(f, "%d %d %f\n", i,j , distanceTable[i][j]);
		}
		fprintf(f, "\n");
	}
	fclose(f);
	waitKey();
}

void PedestrianClustering()
{
	double centre[4] = { 0 };
	centre[0] = distanceTable[rand() % 34][rand() % 34];
	centre[1] = distanceTable[rand() % 34][rand() % 34];
	centre[2] = distanceTable[rand() % 34][rand() % 34];
	centre[3] = distanceTable[rand() % 34][rand() % 34];
}


Mat lab4TransformataDistanta(Mat imagine){

	Mat destinatie = imagine.clone();

	int masca[3][3] = { { 3, 2, 3 }, { 2, 0, 2 }, { 3, 2, 3 } };

	for (int i = 1; i < imagine.rows - 1; i++){
		for (int j = 1; j < imagine.cols - 1; j++){

			int valoareMinima = 300;

			for (int k = -1; k < 2; k++){
				for (int t = -1; t < 2; t++){
					if (destinatie.at<uchar>(i + k, j + t) + masca[k + 1][t + 1] < valoareMinima){
						valoareMinima = destinatie.at<uchar>(i + k, j + t) + masca[k + 1][t + 1];
					}
				}
			}
			destinatie.at<uchar>(i, j) = valoareMinima;
		}
	}

	for (int i = imagine.rows - 2; i > 0; i--){
		for (int j = imagine.cols - 2; j > 0; j--){

			int valoareMinima = destinatie.at<uchar>(i, j);
			for (int k = -1; k < 2; k++){
				for (int t = -1; t < 2; t++){
					if (destinatie.at<uchar>(i + k, j + t) + masca[k + 1][t + 1] < valoareMinima){
						valoareMinima = destinatie.at<uchar>(i + k, j + t) + masca[k + 1][t + 1];
					}
				}
			}
			destinatie.at<uchar>(i, j) = valoareMinima;
		}
	}

	return destinatie;
}

double lab4PatternRecognition(Mat imagine1, Mat imagine2){

	Mat imagineTransformata = lab4TransformataDistanta(imagine1);
	Mat imagineSursa = imagine2;

	float suma = 0;
	int nrPuncteCoordonate = 0;
	int media = 0;

	for (int i = 0; i < imagineTransformata.rows - 1; i++){
		for (int j = 0; j < imagineTransformata.cols - 1; j++){
			if (imagineSursa.at<uchar>(i, j) == 0){
				suma += imagineTransformata.at<uchar>(i, j) / 2;
				nrPuncteCoordonate++;
			}
		}
	}
	return suma / nrPuncteCoordonate;
}


void Cluster()
{
	Mat img = imread("Images/points2.bmp", IMREAD_GRAYSCALE);
	Mat dst(500, 500, CV_8UC3);
	int points[4000][3] = { 0 }, k = -1, centre[3] = { 0 }, min = 999999;
	int mi = 0, mj = 0;
	//get all points
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) == 0)
			{
				//k is number of points
				k++;
				// 0 means x coord j
				// 1 means y coord i
				// 2 means the cluster number init 0
				points[k][0] = j;
				points[k][1] = i;
				points[k][2] = 0;


			}

		}
	centre[0] = rand() % k;
	centre[1] = rand() % k;
	centre[2] = rand() % k;
	//centre[3] = rand() % k;

	int plusPoints = 0;
	//iterate through each cluster
	bool changed = true;


	for (int l = 1; l <= 20; l++)
	{
			for (int i = 0; i < k; i++)
			{
				min = 999999;
				//check for every point the most appropriate cluster
				for (int j = 0; j < 3; j++)
				{   
					//calcultate distance between center and point
					int val = sqrt(pow(abs(points[i][0] - points[centre[j]][0]), 2) + pow(abs(points[i][1] - points[centre[j]][1]), 2));
					if (val < min)
					{

						min = val;
						points[i][2] = j;
					}
				}
			}

			// check the number of points of each cluster
			for (int j = 0; j < 3; j++)
			{
				mi = 0;
				mj = 0;
				int nrp = 0;

				for (int i = 0; i < k; i++)
				{
					//if point belongs to cluster j
					if (points[i][2] == j)
					{

						// sum all x coord and all y and count points
						mi += points[i][1];
						nrp++;
						mj += points[i][0];

					}
				}
				if (mi / nrp != points[k + plusPoints][1] || mj / nrp != points[k + plusPoints][0]) changed = true;

				plusPoints++;
				points[k + plusPoints][1] = mi / nrp;
				points[k + plusPoints][0] = mj / nrp;
				centre[j] = k + plusPoints - 1;
			}
	
	}


	Vec3b colors[4];
	for (int i = 0; i < 3; i++)
		colors[i] = { (uchar)(rand() % 255), (uchar)(rand() % 255), (uchar)(rand() % 255) };

	for (int i = 0; i < k; i++)
	{
		dst.at<Vec3b>(points[i][1], points[i][0]) = colors[points[i][2]];
	}

	imshow("DT", dst);
	waitKey();
}



int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Clustering \n");
		printf(" 11 - Pedestrian\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				Cluster();
				break;
			case 11:
				createPedestriansTable();
				break;
		}
	}
	while (op!=0);
	return 0;
}