#ifndef __CLUTSEG__
#define __CLUTSEG__

#include "ocv_headers.h"

#define offsetHS 0
#define offsetUV 1

struct colorSegment
{
  uchar val[3];
};

struct valueXY
{
  float x;
  float y;
};

struct mouseparam
{
  valueXY start;
  valueXY now;
  bool drawing;
  bool done;
};

struct plotparam
{
  float margin;
  valueXY a;
  valueXY b;
  valueXY c;

  // axis value
  valueXY min;
  valueXY max;

  // scale value in plot
  valueXY scale;
};

// init and configuration
void initCalibration(valueXY *calibHS, valueXY *calibUV, int segNumHS, int segNumUV, float margin = 10.0f);
// void calculateLUT();
// cvtHSV(cv::Mat img);
// cvtYUV(cv::Mat img);
// void calibMouseCallback(int event, int x, int y, int flags, void *param);

// display
// int checkGradient(cv::Point Pt1, cv::Point Pt2);
// valueXY mouse2value(valueXY mousePos, plotparam param);
// void plot(cv::Mat& img, cv::Mat& plot, plotparam param, int offset);
// void plotLine(valueXY *valueCalib, cv::Mat& plot, plotparam param, int numPts);
// void drawingLine(cv::Mat& plot, valueXY *valCalib, plotparam param, char key);

// do calibration
void Calibrate(cv::Mat& img, char key);

// segmenting
// uchar lineardiscriminator(valueXY img, valueXY *valCalib, int idxSeg);
void segmentImg(cv::Mat& imgInput, cv::Mat& imgOutput, colorSegment *color);
void segmentImg(cv::Mat& imgInput, cv::Mat& imgOutput, int discriminator);

#endif