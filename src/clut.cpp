#include "clut.h"

struct mouseparam mice;
int calibration = 0;
bool destroyed = false;
bool callbackStatus = false;
cv::Mat blankplot;
cv::Mat updatePlotHS, updatePlotUV;
cv::Mat HSVImg;
cv::Mat YUVImg;

float Margin = 0.f;
int numLinearDiscriminatorPtsHS = 0;
int numLinearDiscriminatorPtsUV = 0;
int totalSeg;
int segUV;
int segHS;

plotparam paramHS = {0};
plotparam paramUV = {0};
float *discr;

valueXY *valCalibHS;
valueXY *valCalibUV;

int lineCalib = 0;

void initCalibration(valueXY *calibHS, valueXY *calibUV, int segNumHS, int segNumUV, float margin)
{
  Margin = margin;
  blankplot = cv::Mat(400, 400, CV_8UC3, cv::Scalar(255,255,255));
  discr = (float *)malloc(4*sizeof(float));
  
  // set N points for linear discriminator
  numLinearDiscriminatorPtsHS = segNumHS * 4 * 2;
  numLinearDiscriminatorPtsUV = segNumUV * 4 * 2;

  // set total segmentation
  totalSeg = segNumHS + segNumUV;
  segUV = segNumUV;
  segHS = segNumHS;

  // set calibration value pointer
  valCalibHS = calibHS;
  valCalibUV = calibUV;

  // set paramHS
  paramHS.margin = Margin;
  paramHS.a = {Margin, Margin};
  paramHS.b = {Margin, (float)blankplot.rows - Margin};
  paramHS.c = {(float)blankplot.cols - Margin, (float)blankplot.rows - Margin};
  paramHS.min = {0,0};
  paramHS.max = {179,255};

  paramHS.scale.y = (paramHS.b.y - paramHS.a.y)/(paramHS.max.y - paramHS.min.y);
  paramHS.scale.x = (paramHS.c.x - paramHS.b.x)/(paramHS.max.x - paramHS.min.x);

  // set paramUV
  paramUV.margin = Margin;
  paramUV.a = paramHS.a;
  paramUV.b = paramHS.b;
  paramUV.c = paramHS.c;
  paramUV.min = {0,0};
  paramUV.max = {255,255};

  paramUV.scale.y = (paramUV.b.y - paramUV.a.y)/(paramUV.max.y - paramUV.min.y);
  paramUV.scale.x = (paramUV.c.x - paramUV.b.x)/(paramUV.max.x - paramUV.min.x);
  
  cv::namedWindow("plot_YUV", cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
  cv::namedWindow("plot_HSV", cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
}

cv::Point2f checkGradient(const cv::Point2f& Pt1, const cv::Point2f& Pt2)
{
  const valueXY result = {Pt1.x - Pt2.x, Pt2.y - Pt1.y};
  const float dist = 5.0f;
  const cv::Point2f midPt = Pt1 - (cv::Point2f(Pt1.x - Pt2.x, Pt1.y - Pt2.y))/2;
  if(result.y < 0 && result.x < 0)
    return midPt + cv::Point2f(dist,dist);
  else if(result.y > 0 && result.x > 0)
    return midPt - cv::Point2f(dist,dist);
  else if(result.y < 0 && result.x > 0)
    return midPt + cv::Point2f(-dist,dist);
  else if(result.y > 0 && result.x < 0)
    return midPt - cv::Point2f(-dist,dist);
  else if(result.y == 0 && result.x < 0)
    return midPt + cv::Point2f(0,dist);
  else if(result.y == 0 && result.x >= 0)
    return midPt + cv::Point2f(0,-dist);
  else if(result.y < 0 && result.x == 0)
    return midPt + cv::Point2f(dist,0);
  else //if(result.y <= 0 && result.x == 0)
    return midPt + cv::Point2f(-dist,0);
}

valueXY mouse2value(valueXY mousePos, plotparam param)
{
  valueXY result;
  result.x = (mousePos.x - param.b.x)/param.scale.x;
  result.y = (param.b.y - mousePos.y)/param.scale.y;
  return result;
}

void plotLine(valueXY *valueCalib, cv::Mat& plot, plotparam param, int numPts)
{
  cv::Point2f Pt1, Pt2;
  for(int i = 0; i < numPts; i+=2)
  {
    Pt1 = cv::Point2f(param.b.x + (valueCalib[i+0].x * param.scale.x), param.b.y - (valueCalib[i+0].y * param.scale.y));
    Pt2 = cv::Point2f(param.b.x + (valueCalib[i+1].x * param.scale.x), param.b.y - (valueCalib[i+1].y * param.scale.y));
    cv::line(plot, Pt1, Pt2, cv::Scalar(0,0,255));
    cv::circle(plot, checkGradient(Pt1, Pt2), 3, cv::Scalar(0,155,0), -1);
  }
}

void plot(cv::Mat& img, cv::Mat& plot, plotparam param, int offset)
{
  cv::Mat tempImg;
  valueXY valueTempImg;
  cv::Point2f pt;
  const int imgSize = img.rows * img.cols * img.channels();
  if(offset == offsetHS)
    cv::cvtColor(img, tempImg, cv::COLOR_BGR2HSV);
  else
    cv::cvtColor(img, tempImg, cv::COLOR_BGR2YUV);
  
  for(int j = 0;j < imgSize; j+=3)
  {
    valueTempImg = {(float)tempImg.data[j+0+offset], (float)tempImg.data[j+1+offset]};
    pt = cv::Point2f(param.b.x + (valueTempImg.x * param.scale.x), param.b.y - (valueTempImg.y * param.scale.y));
    cv::circle(plot, pt, 1, cv::Scalar(img.data[j+0], img.data[j+1], img.data[j+2]), -1);
  }
}

void calibMouseCallback(int event, int x, int y, int flags, void *param)
{
  if(event == cv::EVENT_LBUTTONDOWN)
  {
    mice.start = {(float)x, (float)y};
    mice.drawing = true;
  }
  else if(event == cv::EVENT_MOUSEMOVE)
  {
    mice.now = {(float)x, (float)y};
  }
  else if(event == cv::EVENT_LBUTTONUP)
  {
    mice.now = {(float)x, (float)y};
    mice.drawing = false;
    mice.done = true;
  }
  else if(event == cv::EVENT_RBUTTONDOWN)
  {

  }
}

void drawingLine(cv::Mat& plot, valueXY *valCalib, plotparam param, const char& key)
{
  if(key == 'e')
  {
    if(lineCalib != 0)
      lineCalib--;
    else
      lineCalib = 7;
  }
  else if(key == 'f')
  {
    // flip line pt
    valueXY temp;
    int idx;
    if(lineCalib != 0)
      idx = lineCalib - 1;
    else
      idx = 7;
    temp = valCalib[2*idx];
    valCalib[2*idx] = valCalib[2*idx+1];
    valCalib[2*idx+1] = temp;
  }
  
  if(mice.drawing)
  {
    const cv::Point Pt1(mice.start.x, mice.start.y);
    const cv::Point Pt2(mice.now.x, mice.now.y);
    cv::line(plot, Pt1, Pt2, cv::Scalar(0,0,255));
    cv::circle(plot, checkGradient(Pt1, Pt2), 3, cv::Scalar(0,155,0), -1);
    // cv::circle(plot, Pt1, 3, cv::Scalar(155,0,0), -1);
  }
  if(mice.done)
  {
    valCalib[2*lineCalib] = mouse2value(mice.start, param);
    valCalib[2*lineCalib+1] = mouse2value(mice.now, param);
    mice.done = false;
    lineCalib++;
    if(lineCalib == 8)
      lineCalib = 0;
  }
}

void Calibrate(cv::Mat& img, char key)
{
  if(key == 'c')
  {
    lineCalib = 0;
    callbackStatus = false;
    destroyed = false;
    if(calibration == 0 || calibration == 2)
      calibration = 1;
    else
      calibration = 0;
  }
  else if(key == 'v')
  {
    lineCalib = 0;
    callbackStatus = false;
    destroyed = false;
    if(calibration == 0 || calibration == 1)
      calibration = 2;
    else
      calibration = 0;
  }

  if(calibration == 1)
  {
    if(!callbackStatus)
    {
      cv::namedWindow("plot_HSV", cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
      cv::setMouseCallback("plot_HSV", calibMouseCallback);
      callbackStatus = true;
    }
    // draw plot
    blankplot.copyTo(updatePlotHS);
    plot(img, updatePlotHS, paramHS, offsetHS);
    plotLine(valCalibHS, updatePlotHS, paramHS, numLinearDiscriminatorPtsHS);

    // get mouse to calibration
    drawingLine(updatePlotHS, valCalibHS, paramHS, key);

    cv::imshow("plot_HSV", updatePlotHS);
    if(!destroyed)
    {
      cv::destroyWindow("plot_YUV");
      destroyed = true;
    }
  }
  else if(calibration == 2)
  {
    if(!callbackStatus)
    {
      cv::namedWindow("plot_YUV", cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
      cv::setMouseCallback("plot_YUV", calibMouseCallback);
      callbackStatus = true;
    }
    // draw plot
    blankplot.copyTo(updatePlotUV);
    plot(img, updatePlotUV, paramUV, offsetUV);
    plotLine(valCalibUV, updatePlotUV, paramUV, numLinearDiscriminatorPtsUV);

    // get mouse to calibration
    drawingLine(updatePlotUV, valCalibUV, paramUV, key);

    cv::imshow("plot_YUV", updatePlotUV);

    if(!destroyed)
    {
      cv::destroyWindow("plot_HSV");
      destroyed = true;
    }
  }
  else
  {
    if(!destroyed)
    {
      cv::destroyWindow("plot_YUV");
      cv::destroyWindow("plot_HSV");
      destroyed = true;
    }
  }
  
}

uchar lineardiscriminator(const valueXY& img, valueXY *valCalib, const int idxSeg)
{
  for(int i = 0; i < 4; i++)
  {
    const int idx1 = 2*i+4*idxSeg;
    const int idx2 = 2*i+1+4*idxSeg;
    const valueXY valResult = {valCalib[idx1].x - valCalib[idx2].x, valCalib[idx1].y - valCalib[idx2].y};
    if(valResult.y == 0 && valResult.x < 0)
      discr[i] = img.y > valCalib[idx1].y ? -1 : 1;
    else if(valResult.y == 0 && valResult.x >= 0)
      discr[i] = img.y < valCalib[idx1].y ? -1 : 1;
    else if(valResult.y < 0 && valResult.x == 0)
      discr[i] = img.x > valCalib[idx1].x ? -1 : 1;
    else if(valResult.y >= 0 && valResult.x == 0)
      discr[i] = img.x < valCalib[idx1].x ? -1 : 1;
    else
      discr[i] = (img.x - valCalib[idx1].x)/(valCalib[idx2].x - valCalib[idx1].x) 
                - (img.y - valCalib[idx1].y)/(valCalib[idx2].y - valCalib[idx1].y);
    if(discr[i] < 0)
      return 0;
  }
  return 255;
}

void segmentImg(cv::Mat& imgInput, cv::Mat& imgOutput, colorSegment *color)
{
  const int imgInputSize = imgInput.rows * imgInput.cols * imgInput.channels();
  imgOutput = cv::Mat(imgInput.rows, imgInput.cols, CV_8UC3, cv::Scalar(0,0,0));
  cv::cvtColor(imgInput, HSVImg, cv::COLOR_BGR2HSV);
  cv::cvtColor(imgInput, YUVImg, cv::COLOR_BGR2YUV);

  for(int i = 0; i < imgInputSize; i+=3)
  {
    for(int idx1 = 0; idx1 < segHS; idx1++)
    {
      const uchar res = lineardiscriminator({(float)HSVImg.data[i], (float)HSVImg.data[i+1]}, valCalibHS, idx1);
      if(imgOutput.data[i] == 0)
        imgOutput.data[i] = res != 0 ? color[idx1].val[0] : 0;
      if(imgOutput.data[i+1] == 0)
        imgOutput.data[i+1] = res != 0 ? color[idx1].val[1] : 0;
      if(imgOutput.data[i+2] == 0)
        imgOutput.data[i+2] = res != 0 ? color[idx1].val[2] : 0;
    }
    for(int idx2 = 0; idx2 < segUV; idx2++)
    {
      const int idx2color = idx2 + segHS;
      const uchar res = lineardiscriminator({(float)YUVImg.data[i+1], (float)YUVImg.data[i+2]}, valCalibUV, idx2);
      if(imgOutput.data[i] == 0)
        imgOutput.data[i] = res != 0 ? color[idx2color].val[0] : 0;
      if(imgOutput.data[i+1] == 0)
        imgOutput.data[i+1] = res != 0 ? color[idx2color].val[1] : 0;
      if(imgOutput.data[i+2] == 0)
        imgOutput.data[i+2] = res != 0 ? color[idx2color].val[2] : 0;
    }
  }
}

void segmentImg(cv::Mat& imgInput, cv::Mat& imgOutput, const int discriminator)
{
  const int imgInputSize = imgInput.rows * imgInput.cols * imgInput.channels();
  imgOutput = cv::Mat(imgInput.rows, imgInput.cols, CV_8UC3, cv::Scalar::all(0));
  valueXY *tempPtr; 
  uchar *imgData;
  int offset1;
  int idx;
  if(discriminator < segHS)
  {
    cv::cvtColor(imgInput, HSVImg, cv::COLOR_BGR2HSV);
    tempPtr = valCalibHS;
    imgData = HSVImg.data;
    offset1 = 0;
    idx = discriminator;
  }
  else
  {
    cv::cvtColor(imgInput, YUVImg, cv::COLOR_BGR2YUV);
    tempPtr = valCalibUV;
    imgData = YUVImg.data;
    offset1 = 1;
    idx = discriminator - segHS;
  }

  for(int i = 0; i < imgInputSize; i+=3)
  {
    imgOutput.data[i] = lineardiscriminator({(float)imgData[i+offset1], (float)imgData[i+1+offset1]}, tempPtr, idx);
    imgOutput.data[i+1] = imgOutput.data[i];
    imgOutput.data[i+2] = imgOutput.data[i];
  }
}