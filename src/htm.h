#ifndef HTM_H
#define HTM_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "math.h"

#include "types.h"
#include "segment.h"
#include "bitarray.h"

typedef struct htm_params_t_ {
    u8 permanence_threshold;
    u8 segment_spiking_threshold;

    u8 perm_increment;
    u8 perm_decrement;
    u8 perm_decay; // how much to decay predicted but inactive cells 
    // TODO: maybe add a decay period?
} htm_params_t;

typedef struct extended_htm_params_t_ {
    u8 feedforward_permanence_threshold;
    u8 context_permanence_threshold;

    u8 feedforward_activation_threshold;
    u8 context_activation_threshold;

    u8 min_active_cells;
} extended_htm_params_t;

void htm_prediction_integrate_context(
    u32* num_spiking_segments,
    segment_t* context_pointer, u32* active,
    enum segment_index_type index_type, u8 num_segments,
    u8 permanence_threshold, u8 segment_spiking_threshold
);

void htm_activate(
    u32* active_columns, 
    u32* activity_bitarrays, u32* prediction_bitarrays, 
    u32 cols, u32 cells
);

void htm_learning_pick_winner_cell(
    u32* winning_cell, u32* winning_connection_count, 
    segment_t* context_pointer, 
    u32 segments, u32 cells
);

void htm_learning_adjust_permanences(
    u32* incident_activity_prev, segment_t* context_pointer,
    u32 winning_cell, u32 winning_connection_count,
    u32 active_bitarray, u32 pred_bitarray,
    enum segment_index_type index_type, u8 num_segments,
    htm_params_t htm_p, u32 cell
);

#endif
