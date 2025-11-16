#ifndef FEATURE_LAYER_H
#define FEATURE_LAYER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "segment.h"
#include "bitarray.h"
#include "htm.h"

typedef struct location_layer_params_t_ location_layer_params_t;
typedef struct location_layer_t_ location_layer_t;

typedef struct feature_layer_params_t_ {
    u16 cols;
    u16 cells; // per columns

    u8 feature_segments; // per cell
    u8 location_segments; // per cell

    htm_params_t htm;
} feature_layer_params_t;

typedef struct l4_segment_tensor_ {
    // the feedforward segments are already taken care of in the pooling layer, transmitted via "active_columns" in activate()
    segment_t* feature_context; // of size cols * cells * feature_segments * CONNECTIONS_PER_SEGMENT * sizeof(segment_data_t)
    segment_t* location_context; // of size cols * cells * feature_segments * CONNECTIONS_PER_SEGMENT * sizeof(segment_data_t)
} l4_segment_tensor;

typedef struct feature_layer_t_ {
    l4_segment_tensor in_segments; // of shape (#cols, #cells, #segments)

    u32* predicted; // of shape (#minicols) where each entry is a bitarray representing the cells in each col [sparse x sparse]
    u32* active; // of shape (#minicols) where each entry is a bitarray representing the cells in each col [sparse x sparse]

    u32* active_prev; // same as active. Represents the last state, used for learning

    feature_layer_params_t p;
} feature_layer_t;

void init_l4_segment_tensor(feature_layer_t* net, location_layer_params_t location_layer_p);

void init_feature_layer(feature_layer_t* network, feature_layer_params_t p, location_layer_params_t location_layer_p);

void feature_layer_predict(feature_layer_t* net, location_layer_t* location_net);

void feature_layer_activate(feature_layer_t* net, u8* active_columns, location_layer_t* location_net);

#endif
