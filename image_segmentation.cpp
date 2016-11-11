#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/legacy/legacy.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>

using namespace cv;
using namespace std;

namespace cv
{
    struct ImageFeatures
    {
        vector<KeyPoint> keypoints;
        Mat descriptors;
    };

    struct MatchesInfo
    {
        vector<DMatch> matches;
        vector<DMatch> good_matches;
        Mat H;
    };
}

// segment image src into m*n region of interests.
void imageSegment(Mat& src_img, int m, int n, vector<Mat>& ceil_img)
{
    int height = src_img.rows;
    int width  = src_img.cols;

    int ceil_height = height/m;
    int ceil_width  = width/n;

    Mat roi_img;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            Rect rect(j*ceil_width, i*ceil_height, ceil_width, ceil_height);
            //src_img(rect).copyTo(roi_img);
            roi_img = src_img(rect).clone();
            ceil_img.push_back(roi_img);
        }
    }
}


void orbFeatExtract(Mat& img, ImageFeatures& features, Mat& featImg)
{
    // Detect the keypoints using ORB Detector
    int inums = 200;
    OrbFeatureDetector detector(inums);
    detector.detect(img, features.keypoints);
    drawKeypoints(img, features.keypoints, featImg, Scalar::all(255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Calculate descriptors (feature vectors)
    OrbDescriptorExtractor extractor;
    extractor.compute(img, features.keypoints, features.descriptors);
}


void bfMatcher(ImageFeatures& feat1, ImageFeatures& feat2, MatchesInfo& pairwise_matches)
{
    BruteForceMatcher< L2<float> > matcher;
    matcher.match(feat1.descriptors, feat2.descriptors, pairwise_matches.matches);

    double max_dist = 0, min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < feat1.descriptors.rows; i++ )
    {
        double dist = pairwise_matches.matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;

        if( dist < 3 * min_dist )
            pairwise_matches.good_matches.push_back( pairwise_matches.matches[i]);
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Localize the object from img_1 in img_2
    vector<Point2f> obj, scene;

    for( int i = 0; i < pairwise_matches.good_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( feat1.keypoints[ pairwise_matches.good_matches[i].queryIdx ].pt );
      scene.push_back( feat2.keypoints[ pairwise_matches.good_matches[i].trainIdx ].pt );
    }

    pairwise_matches.H = findHomography( obj, scene, CV_RANSAC );
}



int main (int argc, char **argv)
{
    if(argc < 3)
    {
        cout << "Please input two image!\n";
        cout << "Input such as: <app> <img1.jpg> <img2.jpg>\n";
        return -1;
    }

    int m = 2, n= 2;

    Mat image1 = imread(argv[1], 1);
    Mat image2 = imread(argv[2], 1);
    vector<Mat> roi_img1, roi_img2;

    imshow("origin #1", image1);
    imshow("origin #2", image2);

    imageSegment(image1, m, n, roi_img1);
    imageSegment(image2, m, n, roi_img2);

/*
    imshow("roi_img #10", roi_img1[0]);
    imshow("roi_img #11", roi_img1[1]);
    imshow("roi_img #12", roi_img1[2]);
    imshow("roi_img #13", roi_img1[3]);

    imshow("roi_img #20", roi_img2[0]);
    imshow("roi_img #21", roi_img2[1]);
    imshow("roi_img #22", roi_img2[2]);
    imshow("roi_img #23", roi_img2[3]);
*/
    vector<ImageFeatures> features1(m*n);
    vector<ImageFeatures> features2(m*n);
    vector<Mat> featImage1(m*n);
    vector<Mat> featImage2(m*n);
    for(int i=0; i < m*n; i++)
    {
        orbFeatExtract(roi_img1[i], features1[i], featImage1[i]);
        orbFeatExtract(roi_img2[i], features2[i], featImage2[i]);
    }

    imshow("featImg #10", featImage1[0]);
    imshow("featImg #11", featImage1[1]);
    imshow("featImg #12", featImage1[2]);
    imshow("featImg #13", featImage1[3]);

    imshow("featImg #20", featImage2[0]);
    imshow("featImg #21", featImage2[1]);
    imshow("featImg #22", featImage2[2]);
    imshow("featImg #23", featImage2[3]);


    vector<MatchesInfo> pairwish_matches(m*n);
    vector<Mat> img_matches(m*n);
    vector<Mat> good_matches(m*n);
    for(int i=0; i < m*n; i++)
    {
        //flannMatcher(features1[i], features2[i], pairwish_matches[i]);
        bfMatcher(features1[i], features2[i], pairwish_matches[i]);

        drawMatches(roi_img1[i], features1[i].keypoints, roi_img2[i], features2[i].keypoints,
                    pairwish_matches[i].matches, img_matches[i]);

        // only draw good matches.
        drawMatches(roi_img1[i], features1[i].keypoints, roi_img2[i], features2[i].keypoints,
                    pairwish_matches[i].good_matches, good_matches[i], Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    }

    imshow("matchImg #0", img_matches[0]);
    imshow("matchImg #1", img_matches[1]);
    imshow("matchImg #2", img_matches[2]);
    imshow("matchImg #3", img_matches[3]);

    imshow("goodMatch #0", good_matches[0]);
    imshow("goodMatch #1", good_matches[1]);
    imshow("goodMatch #2", good_matches[2]);
    imshow("goodMatch #3", good_matches[3]);

    waitKey(0);
    return 0;
}
