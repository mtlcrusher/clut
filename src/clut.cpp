#include "clut.h"

struct mouseparam mice;
int calibration = 0;
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
float discr[4] = {0};

valueXY *valCalibHS;
valueXY *valCalibUV;

int lineCalib = 0;

void initCalibration(valueXY *calibHS, valueXY *calibUV, int segNumHS, int segNumUV, float margin)
{
  Margin = margin;
  blankplot = cv::Mat(400, 400, CV_8UC3, cv::Scalar(255,255,255));

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
  
  cv::namedWindow("plot_YUV", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("plot_HSV", cv::WINDOW_AUTOSIZE);
}

cv::Point2f checkGradient(cv::Point2f Pt1, cv::Point2f Pt2)
{
  valueXY result = {Pt1.x - Pt2.x, Pt2.y - Pt1.y};
  float dist = 5.0f;
  cv::Point2f midPt = Pt1 - (cv::Point2f(Pt1.x - Pt2.x, Pt1.y - Pt2.y))/2;
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

cv::Point2f value2point(valueXY value, plotparam param)
{
  return cv::Point2f(param.b.x + (value.x * param.scale.x), param.b.y - (value.y * param.scale.y));
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
  valueXY *tempPtr = valueCalib;
  cv::Point2f Pt1, Pt2;
  while(tempPtr - valueCalib < numPts)
  {
    Pt1 = value2point(tempPtr[0], param);
    Pt2 = value2point(tempPtr[1], param);
    cv::line(plot, Pt1, Pt2, cv::Scalar(0,0,255));
    cv::circle(plot, checkGradient(Pt1, Pt2), 3, cv::Scalar(0,155,0), -1);
    tempPtr+=2;
  }
}

void plot(cv::Mat& img, cv::Mat& plot, plotparam param, int offset)
{
  cv::Mat tempImg;
  valueXY valueTempImg;
  cv::Point2f pt;
  int imgSize = img.rows * img.cols * img.channels();
  if(offset == offsetHS)
    cv::cvtColor(img, tempImg, cv::COLOR_BGR2HSV);
  else
    cv::cvtColor(img, tempImg, cv::COLOR_BGR2YUV);
  
  for(int j = 0;j < imgSize; j+=3)
  {
    valueTempImg = {(float)tempImg.data[j+0+offset], (float)tempImg.data[j+1+offset]};
    pt = value2point(valueTempImg, param);
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

void drawingLine(cv::Mat& plot, valueXY *valCalib, plotparam param, char key)
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
    cv::Point Pt1(mice.start.x, mice.start.y);
    cv::Point Pt2(mice.now.x, mice.now.y);
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
    if(calibration == 0 || calibration == 2)
      calibration = 1;
    else
      calibration = 0;
  }
  else if(key == 'v')
  {
    lineCalib = 0;
    callbackStatus = false;
    if(calibration == 0 || calibration == 1)
      calibration = 2;
    else
      calibration = 0;
  }

  if(calibration == 1)
  {
    if(!callbackStatus)
    {
      cv::namedWindow("plot_HSV", cv::WINDOW_AUTOSIZE);
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
    cv::destroyWindow("plot_YUV");
  }
  else if(calibration == 2)
  {
    if(!callbackStatus)
    {
      cv::namedWindow("plot_YUV", cv::WINDOW_AUTOSIZE);
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
    cv::destroyWindow("plot_HSV");
  }
  else
  {
    cv::destroyWindow("plot_YUV");
    cv::destroyWindow("plot_HSV");
  }
  
}

uchar lineardiscriminator(valueXY img, valueXY *valCalib, int idxSeg)
{
  for(int i = 0; i < 4; i++)
  {
    valueXY valCalib1 = valCalib[2*i+4*idxSeg];
    valueXY valCalib2 = valCalib[2*i+1+4*idxSeg];
    valueXY valResult = {valCalib1.x - valCalib2.x, valCalib1.y - valCalib2.y};
    if(valResult.y == 0 && valResult.x < 0)
      discr[i] = img.y > valCalib1.y ? -1 : 1;
    else if(valResult.y == 0 && valResult.x >= 0)
      discr[i] = img.y < valCalib1.y ? -1 : 1;
    else if(valResult.y < 0 && valResult.x == 0)
      discr[i] = img.x > valCalib1.x ? -1 : 1;
    else if(valResult.y >= 0 && valResult.x == 0)
      discr[i] = img.x < valCalib1.x ? -1 : 1;
    else
      discr[i] = (img.x - valCalib1.x)/(valCalib2.x - valCalib1.x) 
                - (img.y - valCalib1.y)/(valCalib2.y - valCalib1.y);
    if(discr[i] < 0)
      return 0;
  }
  return 1;
}

void segmentImg(cv::Mat imgInput, cv::Mat& imgOutput, colorSegment *color)
{
  int imgInputSize = imgInput.rows * imgInput.cols * imgInput.channels();
  imgOutput = cv::Mat(imgInput.rows, imgInput.cols, CV_8UC3, cv::Scalar(0,0,0));
  cv::cvtColor(imgInput, HSVImg, cv::COLOR_BGR2HSV);
  cv::cvtColor(imgInput, YUVImg, cv::COLOR_BGR2YUV);

  for(int i = 0; i < imgInputSize; i+=3)
  {
    valueXY hs = {(float)HSVImg.data[i], (float)HSVImg.data[i+1]};
    valueXY uv = {(float)YUVImg.data[i+1], (float)YUVImg.data[i+2]};
    for(int idx1 = 0; idx1 < segHS; idx1++)
    {
      uchar res = lineardiscriminator(hs, valCalibHS, idx1);
      imgOutput.data[i] = imgOutput.data[i] != 0 ? imgOutput.data[i] : color[idx1].val[0] * res;
      imgOutput.data[i+1] = imgOutput.data[i+1] != 0 ? imgOutput.data[i+1] : color[idx1].val[1] * res;
      imgOutput.data[i+2] = imgOutput.data[i+2] != 0 ? imgOutput.data[i+2] : color[idx1].val[2] * res;
    }
    for(int idx2 = 0; idx2 < segUV; idx2++)
    {
      int idx2color = idx2 + segHS;
      uchar res = lineardiscriminator(uv, valCalibUV, idx2);
      imgOutput.data[i] = imgOutput.data[i] != 0 ? imgOutput.data[i] : color[idx2color].val[0] * res;
      imgOutput.data[i+1] = imgOutput.data[i+1] != 0 ? imgOutput.data[i+1] : color[idx2color].val[1] * res;
      imgOutput.data[i+2] = imgOutput.data[i+2] != 0 ? imgOutput.data[i+2] : color[idx2color].val[2] * res;
    }
  }
}