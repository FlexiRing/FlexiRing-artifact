#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>


#define NUM_CLASSES   12    
#define FEATURE_DIM   256   
    
#define EPS 1e-8f


extern const float PROTOTYPES[NUM_CLASSES][FEATURE_DIM];

int classify_gesture(float* feature, float* result);

#ifdef __cplusplus
}
#endif

#endif