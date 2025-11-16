#include "htm.h"

void htm_prediction_integrate_context(
    u32* num_spiking_segments,
    segment_t* context_pointer, u32* incident_activity,
    enum segment_index_type index_type, u8 num_segments,
    u8 permanence_threshold, u8 segment_spiking_threshold
) {
    for(u32 seg = 0; seg < num_segments; ++seg) {
        context_pointer->connection_count = 0; // reset this cache

        u32 cell_accumulator = 0;
        for(u32 conn = 0; conn < context_pointer->num_connections; ++conn) {
            segment_data seg_data = context_pointer->connections[conn];

            u32 is_cell_active;
            if(index_type == INTERNAL_OUTPUT_INDEX_TYPE) {
                internal_output_index index = seg_data.index.internal_output;
                is_cell_active = GET_BIT_FROM_PACKED32(incident_activity, index.cell);
            } else if(index_type == EXTERNAL_OUTPUT_INDEX_TYPE) {
                external_output_index index = seg_data.index.external_output;
                is_cell_active = GET_BIT_FROM_PACKED32(incident_activity, index.cell);
            } else if(index_type == FEATURE_INDEX_TYPE) {
                feature_index index = seg_data.index.feature;
                is_cell_active = GET_BIT(incident_activity[index.col], index.cell);
            } else /* if(index_type == LOCATION_INDEX_TYPE) */ {
                location_index index = seg_data.index.location;
                is_cell_active = GET_BIT(incident_activity[index.col], index.cell);
            }

            u32 is_cell_connected = seg_data.permanence >= permanence_threshold;

            cell_accumulator += is_cell_active & is_cell_connected;
        }

        if(cell_accumulator > 255) cell_accumulator = 255;
        context_pointer->connection_count = cell_accumulator;

        if(cell_accumulator > segment_spiking_threshold) *num_spiking_segments += 1;

        context_pointer += 1;
    }
}

void htm_activate(
    u32* active_columns, 
    u32* activity_bitarrays, u32* prediction_bitarrays, 
    u32 cols, u32 cells
) {
    for(u32 col = 0; col < cols; ++col) {
        if(active_columns[col] != 0) {
            u32 act_bitarray = 0; // 000...000 
            for(u32 cell = 0; cell < cells; ++cell) {
                u32 was_predicted = GET_BIT(prediction_bitarrays[col], cell);
                act_bitarray |= (was_predicted << cell);
            }
            // if no cell was activated through predictions, activate all cells in column
            if(act_bitarray == 0) act_bitarray = ~ ((u32) 0); // 111...111

            activity_bitarrays[col] = act_bitarray;
        } else {
            activity_bitarrays[col] = 0; // 000...0000
        }
    } 
}

void htm_learning_pick_winner_cell(
    u32* winning_cell, u32* winning_connection_count, 
    segment_t* context_pointer, 
    u32 segments, u32 cells
) {
    for(u32 cell = 0; cell < cells; ++cell) {
        for(u32 seg = 0; seg < segments; ++seg) {
            if(context_pointer->connection_count > *winning_connection_count) {
                *winning_cell = cell;
                *winning_connection_count = context_pointer->connection_count ;
            }

            context_pointer += 1;
        }
    }
}

void htm_learning_adjust_permanences(
    u32* incident_activity_prev, segment_t* context_pointer,
    u32 winning_cell, u32 winning_connection_count,
    u32 active_bitarray, u32 pred_bitarray,
    enum segment_index_type index_type, u8 num_segments,
    htm_params_t htm_p, u32 cell
) {
    u32 cell_is_predicted = GET_BIT(pred_bitarray, cell);

    for(u32 seg = 0; seg < num_segments; ++seg) {
        /** There are two cases for a reinforcement:
         * 
         * 1. If the column is active, the cell is predicted and the segment was spiking,
         *      we select that segment for reinforcement. That means that we increase the permanences
         *      of active incident cells and decrease the permanences of inactive incident cells on the segment
         * 
         * 2. If the column is active, no cell in the col is predicted and this cell is the winning cell (chosen prior)
         *      we 
         */
        u32 seg_was_spiking = context_pointer->connection_count >= htm_p.segment_spiking_threshold;

        u32 should_reinforce_case1 = active_bitarray != 0 && cell_is_predicted 
            && seg_was_spiking;
        u32 should_reinforce_case2 = active_bitarray != 0 && pred_bitarray == 0 
            && cell == winning_cell 
            && context_pointer->connection_count == winning_connection_count;

        u32 should_reinforce = should_reinforce_case1 && should_reinforce_case2;
        
        if(should_reinforce) {
            for(u32 conn = 0; conn < context_pointer->num_connections; ++conn) {
                segment_data* seg_data = &(context_pointer->connections[conn]);

                // if we don't know how to handle the incident cell index, we default to not handling anything
                u32 incident_cell_was_active = 0; 
                if(index_type == FEATURE_INDEX_TYPE) {
                    feature_index index = seg_data->index.feature;
                    incident_cell_was_active = GET_BIT(incident_activity_prev[index.col], index.cell);
                } else if(index_type == LOCATION_INDEX_TYPE) {
                    location_index index = seg_data->index.location;
                    incident_cell_was_active = GET_BIT(incident_activity_prev[index.col], index.cell);
                }
                
                // we don't care if the cell was connected (perm > thresh), we just reward active and
                // punish inactive for a spiking segment
                if(incident_cell_was_active) {
                    seg_data->permanence = safe_add_u8(
                        seg_data->permanence, 
                        htm_p.perm_increment
                    );
                } else {
                    seg_data->permanence = safe_sub_u8(
                        seg_data->permanence, 
                        htm_p.perm_decrement
                    );
                }
            }
        } 
        
        // If the cell was predicted but ended up not become active, apply a decay to
        // synapses above perm thresh and connected to an active cell
        u32 should_decay = active_bitarray == 0 && cell_is_predicted && seg_was_spiking;
        if(should_decay) {
            for(u32 conn = 0; conn < context_pointer->num_connections; ++conn) {
                segment_data* seg_data = &(context_pointer->connections[conn]);

                // if we don't know how to handle the incident cell index, we default to not handling anything
                u32 incident_cell_was_active = 0; 
                if(index_type == FEATURE_INDEX_TYPE) {
                    feature_index index = seg_data->index.feature;
                    incident_cell_was_active = GET_BIT(incident_activity_prev[index.col], index.cell);
                } else if(index_type == LOCATION_INDEX_TYPE) {
                    location_index index = seg_data->index.location;
                    incident_cell_was_active = GET_BIT(incident_activity_prev[index.col], index.cell);
                }

                if(incident_cell_was_active && seg_data->permanence >= htm_p.permanence_threshold) {
                    seg_data->permanence = safe_sub_u8(
                        seg_data->permanence, 
                        htm_p.perm_decay
                    );
                }
            }
        }

        context_pointer += 1;
    }
}
