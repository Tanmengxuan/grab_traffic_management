# 1 - Model Description 

## Description

The task requires the model to predict as accurate as possible the travel demand at all 
locations (1329 in total) up T+5 time intervals given a sequence of demand data for all locations of arbitrary length T.

This is similar to a language generation problem where the objective is to model the conditional probability of generating new words when given a sequence of length T words as input. The only difference is that 
the travel demand generation model directly predicts the demand at the next time interval while the language generation model has to sample from a probability distribution to decide on the next probable target.

Therefore an Autoregressive Recurrent Neural Network (ARNN) may be used to model the travel demand problem whose aim is to model the future travel pattern given the history tavel demands of users. 

--image of the model--

D_t is a vector representing the demands of users at multiple locations.
At timestep t+1, the ARNN make uses of both the demand information at the previous timestep t (D_t) and its cell state (C_t) as inputs
to predict D_t+1. C_t is an encoded representation of the demand pattern the model has seen so far until timestep t. 

## Making use of location clusters information

Although it is possible to use a single ARNN to model the travel pattern of users at all of the 1329 locations, we want make use of the geohash information given in the raw dataset and use multiple ARNN to capture
travel patterns at different location clusters.

How do we identify the clusters given the geohash information is explained in the [data preprocessing session](https://github.com/Tanmengxuan/grab_traffic_management/tree/master/2_Data_Analysis).

In short, 8 different clusters of locations have been identified and a total of 8 ARNN models are used to model the travel patterns at their corresponding clusters. Each ARNN will be tasked to learn the travel patterns of locations which belong to a certain cluster. 

--image of multiple ARNN--

The 8 ARNN models are being trained simultaneously and the outputs from all 8 models are being merged at the fianl layer and subsequently backpropagated using the root mean square loss function. 
Therefore, the final model is still able to predict the travel demands at all 1329 locations at the same time. 

## Some results

Although the task only requires the model to predict demands up to T+5 time intervals, the ARNN model can 
predict up to any number of time intervals ahead given input sequence of any length T.  

--image of 200-- 

The model is able to capture the demand pattern of a particular location 100 time intervals ahead given an input sequence length of 300 time intervals. 

--image of multiple locations--

Capturing demand patterns at different locations.
