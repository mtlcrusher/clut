#include <iostream>
#include "ocv_headers.h"
#include "clut.h"

using namespace cv;

int main(int argc, char **argv)
{
  char keyB = -1;
  bool segmentedMode = false;
  colorSegment *color = (colorSegment *)malloc(4*sizeof(colorSegment));
  color[0] = {0,211,255};
  color[1] = {35,102,11};
  color[2] = {33,67,101};
  color[3] = {0,0,255};
  printf("color[%d] = %d %d %d\n", 0, color[0].val[0], color[0].val[1], color[0].val[2]);
  VideoCapture camera(0);
  camera.set(3, 320);
  camera.set(4, 240);

  valueXY *valueCalibrationHS = (valueXY *)malloc(16*sizeof(valueXY));
  valueXY *valueCalibrationUV = (valueXY *)malloc(16*sizeof(valueXY));
  float calibval[] = 
  {
    26, 246, 59, 249, 
    65, 154, 55, 251, 
    68, 164, 21, 150, 
    23, 140, 32, 251, 
    28, 115, 62, 121, 
    58, 128, 56, 68, 
    67, 78, 29, 67, 
    31, 120, 39, 53, 
    42, 123, 59, 142, 
    83, 91, 55, 144, 
    85, 99, 61, 73, 
    45, 136, 66, 66, 
    97, 132, 109, 133, 
    113, 110, 103, 137, 
    115, 118, 94, 108, 
    96, 102, 99, 135
  };
  float *cptr = calibval;
  for(int i = 0; i < 16; i+=2)
  {
    valueCalibrationHS[i] = {cptr[0], cptr[1]};
    valueCalibrationHS[i+1] = {cptr[2], cptr[3]};

    valueCalibrationUV[i] = {cptr[32], cptr[33]};
    valueCalibrationUV[i+1] = {cptr[34], cptr[35]};
    cptr+=4;
  }

  for(int i = 0; i < 16; i+=2)
    printf("%.0f, %.0f, %.0f, %.0f,\n", valueCalibrationUV[i].x, valueCalibrationUV[i].y, valueCalibrationUV[i+1].x, valueCalibrationUV[i+1].y);

  Mat camFrame, FrameYUV, FrameHSV;
  Mat HSVImg, YUVImg;
  Mat segmented, outputImg;
  
  initCalibration(valueCalibrationHS, valueCalibrationUV, 2, 2);

  while(1)
  {
    camera >> camFrame;

    if(segmentedMode)
    {
      // segmentImg(camFrame, segmented, color);
      segmentImg(camFrame, segmented, 2);
      camFrame.copyTo(outputImg);
      outputImg &= segmented;
      imshow("segmented", outputImg);
    }
    else
    {
      imshow("camFrame", camFrame);
    }

    // imshow("YUVImg", YUVImg);
    // imshow("HSVImg", HSVImg);
    keyB = waitKey(5);
    Calibrate(camFrame, keyB);
    if(keyB == 'q')
      break;
    else if(keyB == 's')
    {
      if(!segmentedMode)
      {
        destroyWindow("camFrame");
        namedWindow("segmented", WINDOW_AUTOSIZE | WINDOW_OPENGL);
        segmentedMode = true;
      }
      else
      {
        cv::namedWindow("camFrame", WINDOW_AUTOSIZE | WINDOW_OPENGL);
        destroyWindow("segmented");
        segmentedMode = false;
      }
    }
  }

  camera.release();
  free(color);
  free(valueCalibrationHS);
  free(valueCalibrationUV);
  return 0;
}