#include "classifier.h"

static float safe_norm(const float *vec) {
    float acc = 0.f;
    for (int i = 0; i < FEATURE_DIM; ++i) {
        acc += vec[i] * vec[i];
    }
    return sqrtf(acc);
}

int classify_gesture(float *feature, float *result) {
    if (!feature || !result) {
        return -1;
    }

    const float feat_norm = safe_norm(feature);
    if (feat_norm < EPS) {
        return -1;
    }

    int best_class = -1;
    float best_dist = INFINITY;

    for (int cls = 0; cls < NUM_CLASSES; ++cls) {
        const float *proto = PROTOTYPES[cls];
        float dot = 0.f;
        float proto_norm = EPS;

        for (int i = 0; i < FEATURE_DIM; ++i) {
            dot += feature[i] * proto[i];
            proto_norm += proto[i] * proto[i];
        }

        proto_norm = sqrtf(proto_norm);
        const float dist = 1.f - dot / (proto_norm * feat_norm + EPS);
        result[cls] = dist;
        if (dist < best_dist) {
            best_dist = dist;
            best_class = cls;
        }
    }

    return best_class;
}