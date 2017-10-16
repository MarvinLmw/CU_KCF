/*
Author: Christian Bailer
Contact address: Christian.Bailer@dfki.de
Department Augmented Vision DFKI

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#pragma once

//#include <cv.h>
#include <math.h>

#ifndef _OPENCV_RECTTOOLS_HPP_
#define _OPENCV_RECTTOOLS_HPP_
#endif

//#include <sys/time.h>
#include <iostream>

extern int part_time[16];
using namespace std;

namespace RectTools
{
////////////////////////////////////////////////////////////////////
//
//   recttools
//
////////////////////////////////////////////////////////////////////
template <typename t>
inline cv::Vec<t, 2 > center(const cv::Rect_<t> &rect)
{
    return cv::Vec<t, 2 > (rect.x + rect.width / (t) 2, rect.y + rect.height / (t) 2);
}

template <typename t>
inline t x2(const cv::Rect_<t> &rect)
{
    return rect.x + rect.width;
}

template <typename t>
inline t y2(const cv::Rect_<t> &rect)
{
    return rect.y + rect.height;
}

template <typename t>
inline void resize(cv::Rect_<t> &rect, float scalex, float scaley = 0)
{
    if (!scaley)scaley = scalex;
    rect.x -= rect.width * (scalex - 1.f) / 2.f;
    rect.width *= scalex;

    rect.y -= rect.height * (scaley - 1.f) / 2.f;
    rect.height *= scaley;

}

template <typename t>
inline void limit(cv::Rect_<t> &rect, cv::Rect_<t> limit)
{
    if (rect.x + rect.width > limit.x + limit.width)rect.width = (limit.x + limit.width - rect.x);
    if (rect.y + rect.height > limit.y + limit.height)rect.height = (limit.y + limit.height - rect.y);
    if (rect.x < limit.x)
    {
        rect.width -= (limit.x - rect.x);
        rect.x = limit.x;
    }
    if (rect.y < limit.y)
    {
        rect.height -= (limit.y - rect.y);
        rect.y = limit.y;
    }
    if(rect.width<0)rect.width=0;
    if(rect.height<0)rect.height=0;
}

template <typename t>
inline void limit(cv::Rect_<t> &rect, t width, t height, t x = 0, t y = 0)
{
    limit(rect, cv::Rect_<t > (x, y, width, height));
}

template <typename t>
inline int getborder(const cv::Rect_<t > &original, cv::Rect_<t > & limited, cv::Rect_<t > &res)
{
    //cv::Rect_<t > res;
    res.x = limited.x - original.x;
    res.y = limited.y - original.y;
    res.width = x2(original) - x2(limited);
    res.height = y2(original) - y2(limited);
   // if(res.x >= 0 && res.y >= 0 && res.width >= 0 && res.height >= 0)
    //  return 1;
    return 0;//res;
}

int subresize_data(uchar *resdata, const cv::Mat &in, cv::Rect & window, cv::Size & tmp)//const
{
    cv::Rect cutWindow = window;
    RectTools::limit(cutWindow, in.cols, in.rows);
    if (cutWindow.height <= 0 || cutWindow.width <= 0)
	return 1;
    cv::Rect border;
    RectTools::getborder(window, cutWindow, border);
    int left = border.x;
    int top = border.y;
    int right = border.width;
    int bottom = border.height;
	int twidth = tmp.width;
	int theight = tmp.height;

    float dx = window.width/(float)twidth;
    float dy = window.height/(float)theight;
    int offset_r = (cutWindow.x+cutWindow.y*in.cols)*3;

      for (int i=ceil(top/dy);i<theight-ceil(bottom/dy);i++)
      {

          int pos_w = i*twidth*3;
          int pos_r = offset_r+(int)(i*dy-top+0.5)*in.cols*3;
  	  for (int j=0;j<twidth;j++)
  	  {
	    if(j*dx+window.x<=in.cols && cutWindow.y+i*dy<=in.rows && j*dx>=left)
	    {
	      int pos_read = pos_r+((int)(j*dx+0.5)-left)*3;
  	      resdata[pos_w+j*3] = in.data[pos_read];
  	      resdata[pos_w+j*3+1] = in.data[pos_read+1];
  	      resdata[pos_w+j*3+2] = in.data[pos_read+2];
  	    }
  	    else if(right>0 && j*dx+window.x>in.cols && cutWindow.y+i*dy<=in.rows)
  	    {
	      int pos_read = pos_w + (twidth-ceil(right/dx)-1)*3;
	      resdata[pos_w+j*3] = resdata[pos_read];
	      resdata[pos_w+j*3+1] = resdata[pos_read+1];
	      resdata[pos_w+j*3+2] = resdata[pos_read+2];
	    }
            else if(left>0 && j*dx<left && cutWindow.y+i*dy<=in.rows)
	    {
 	      int pos_read = pos_r;
	      resdata[pos_w+j*3] = in.data[pos_read+3];
 	      resdata[pos_w+j*3+1] = in.data[pos_read+4];
	      resdata[pos_w+j*3+2] = in.data[pos_read+5];
	    }
	  }
      }
      if(top>0)
        for(int i=0;i<ceil(top/dy);i++)
          memcpy(&resdata[twidth*i*3], &resdata[(int)(ceil(top/dy)+1)*twidth*3], twidth*3);
      if(bottom>0)
        for(int i=theight-ceil(bottom/dy);i<theight;i++)
          memcpy(&resdata[twidth*i*3], &resdata[theight-(int)ceil(bottom/dy)-1], twidth*3);

  return 0;
}

int subresize_3(cv::Mat &res0, const cv::Mat &in, cv::Rect & window, cv::Size & tmp)//const
{
    cv::Rect cutWindow = window;
    RectTools::limit(cutWindow, in.cols, in.rows);
    if (cutWindow.height <= 0 || cutWindow.width <= 0)
	return 1;
    cv::Rect border;
    RectTools::getborder(window, cutWindow, border);
    int left = border.x;
    int top = border.y;
    int right = border.width;
    int bottom = border.height;

    float dx = window.width/(float)tmp.width;
    float dy = window.height/(float)tmp.height;
    int offset_r = (cutWindow.x+cutWindow.y*in.cols)*3;

      for (int i=ceil(top/dy);i<tmp.height-ceil(bottom/dy);i++)
      {

          int pos_w = i*res0.cols*3;
          int pos_r = offset_r+(int)(i*dy-top+0.5)*in.cols*3;
  	  for (int j=0;j<tmp.width;j++)
  	  {
	    if(j*dx+window.x<=in.cols && cutWindow.y+i*dy<=in.rows && j*dx>=left)
	    {
	      int pos_read = pos_r+((int)(j*dx+0.5)-left)*3;
  	      res0.data[pos_w+j*3] = in.data[pos_read];
  	      res0.data[pos_w+j*3+1] = in.data[pos_read+1];
  	      res0.data[pos_w+j*3+2] = in.data[pos_read+2];
  	    }
  	    else if(right>0 && j*dx+window.x>in.cols && cutWindow.y+i*dy<=in.rows)
  	    {
	      int pos_read = pos_w + (res0.cols-ceil(right/dx)-1)*3;
	      res0.data[pos_w+j*3] = res0.data[pos_read];
	      res0.data[pos_w+j*3+1] = res0.data[pos_read+1];
	      res0.data[pos_w+j*3+2] = res0.data[pos_read+2];
	    }
            else if(left>0 && j*dx<left && cutWindow.y+i*dy<=in.rows)
	    {
 	      int pos_read = pos_r;
	      res0.data[pos_w+j*3] = in.data[pos_read+3];
 	      res0.data[pos_w+j*3+1] = in.data[pos_read+4];
	      res0.data[pos_w+j*3+2] = in.data[pos_read+5];
	    }
	  }
      }
      if(top>0)
        for(int i=0;i<ceil(top/dy);i++)
          memcpy(&res0.data[res0.cols*i*3], &res0.data[(int)(ceil(top/dy)+1)*res0.cols*3], res0.cols*3);
      if(bottom>0)
        for(int i=tmp.height-ceil(bottom/dy);i<tmp.height;i++)
          memcpy(&res0.data[res0.cols*i*3], &res0.data[tmp.height-(int)ceil(bottom/dy)-1], res0.cols*3);

  return 0;
}


int subresize_1(cv::Mat &res0, const cv::Mat &in, cv::Rect & window, cv::Size & tmp)//const
{
    cv::Rect cutWindow = window;
    RectTools::limit(cutWindow, in.cols, in.rows);
    if (cutWindow.height <= 0 || cutWindow.width <= 0)
	return 1;
    cv::Rect border;
    RectTools::getborder(window, cutWindow, border);
    int left = border.x;
    int top = border.y;
    int right = border.width;
    int bottom = border.height;

    float dx = window.width/(float)tmp.width;
    float dy = window.height/(float)tmp.height;
    int offset_r = cutWindow.x+cutWindow.y*in.cols;

      for (int i=ceil(top/dy);i<tmp.height-ceil(bottom/dy);i++)
      {

          int pos_w = i*res0.cols;
          int pos_r = offset_r+(int)(i*dy-top+0.5)*in.cols;
  	  for (int j=0;j<tmp.width;j++)
  	  {
	    if(j*dx+window.x<=in.cols && cutWindow.y+i*dy<=in.rows && j*dx>=left)
	    {
	      int pos_read = pos_r+(int)(j*dx+0.5)-left;
  	      res0.data[pos_w+j] = in.data[pos_read];
  	    }
  	    else if(right>0 && j*dx+window.x>in.cols && cutWindow.y+i*dy<=in.rows)
  	    {
	      int pos_read = pos_w + res0.cols-ceil(right/dx)-1;
	      res0.data[pos_w+j] = res0.data[pos_read];
	    }
            else if(left>0 && j*dx<left && cutWindow.y+i*dy<=in.rows)
	    {
 	      int pos_read = pos_r;
	      res0.data[pos_w+j] = in.data[pos_read+1];
	    }
	  }
      }
      if(top>0)
        for(int i=0;i<ceil(top/dy);i++)
          memcpy(&res0.data[res0.cols*i], &res0.data[(int)ceil(top/dy)*res0.cols], res0.cols);
      if(bottom>0)
        for(int i=tmp.height-ceil(bottom/dy);i<tmp.height;i++)
          memcpy(&res0.data[res0.cols*i], &res0.data[tmp.height-(int)ceil(bottom/dy)-1], res0.cols);

  return 0;
}


inline cv::Mat getGrayImage(cv::Mat img)
{
    cv::cvtColor(img, img, CV_BGR2GRAY);
    img.convertTo(img, CV_32F, 1 / 255.f);
    return img;
}



}



