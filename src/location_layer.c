#include "location_layer.h"

#include "feature_layer.h"

void init_l6_segment_tensor(location_layer_t* net, feature_layer_params_t f_p) {
    // t->cols = cols; t->cells = cells; t->segments = segments;

    net->in_segments.location_context = calloc(net->p.cols * net->p.cells * net->p.location_segments, sizeof(*net->in_segments.location_context));
    net->in_segments.feature_context = calloc(net->p.cols * net->p.cells * net->p.feature_segments, sizeof(*net->in_segments.feature_context));

    u32 tensor_size = net->p.cols * net->p.cells * (net->p.feature_segments + net->p.location_segments) * sizeof(*net->in_segments.location_context);
    printf("-- location layer segment tensor is %u MiB (%u KiB, %u B)\n", tensor_size >> 20, tensor_size >> 10, tensor_size);

    // create feature context by randomly connecting l4 cells together, and connection l6 cells to l4 cells
    segment_t* location_context_pointer = net->in_segments.location_context;
    segment_t* feature_context_pointer = net->in_segments.feature_context;

    for(u32 col = 0; col < net->p.cols; ++col) {
        for(u32 cell = 0; cell < net->p.cells; ++cell) {
            // self location context
            for(u32 seg = 0; seg < net->p.location_segments; ++seg) {
                location_context_pointer->connection_count = 0; // init this cache value to zero
                location_context_pointer->num_connections = unif_rand_range_u32(CONNECTIONS_PER_SEGMENT / 2, CONNECTIONS_PER_SEGMENT); // between 20 and 40 connections

                for(u32 conn = 0; conn < location_context_pointer->num_connections; ++conn) {
                    // connections[conn].index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
                    location_index* index_ptr = &(location_context_pointer->connections[conn].index.location); 
                    index_ptr->col = unif_rand_u32(net->p.cols); // random cell
                    index_ptr->cell = unif_rand_u32(net->p.cells); // random cell

                    location_context_pointer->connections[conn].permanence = unif_rand_u32(255); // random permanence
                }

                location_context_pointer += 1;
            }
        }        
    }

    for(u32 col = 0; col < net->p.cols; ++col) {
        for(u32 cell = 0; cell < net->p.cells; ++cell) {
            // feature context
            for(u32 seg = 0; seg < net->p.feature_segments; ++seg) {
                feature_context_pointer->connection_count = 0; // init this cache value to zero
                feature_context_pointer->num_connections = unif_rand_range_u32(CONNECTIONS_PER_SEGMENT / 2, CONNECTIONS_PER_SEGMENT); // between 20 and 40 connections

                for(u32 conn = 0; conn < feature_context_pointer->num_connections; ++conn) {
                    // connections[conn].index is a union! thus .feature is not a member of a struct, it's just a way to cast to the correct segment type
                    feature_index* index_ptr = &(feature_context_pointer->connections[conn].index.feature); 
                    index_ptr->col = unif_rand_u32(f_p.cols - 1); // random column index
                    index_ptr->cell = unif_rand_u32(f_p.cells - 1); // random cell index

                    feature_context_pointer->connections[conn].permanence = unif_rand_u32(255); // random permanence
                }

                feature_context_pointer += 1;
            }
        }        
    }
}

void init_location_layer(location_layer_t* net, location_layer_params_t p, feature_layer_params_t f_p) {
    net->p = p;

    net->active = calloc(p.cols, sizeof(*net->active));
    net->predicted = calloc(p.cols, sizeof(*net->predicted));

    net->active_prev = calloc(p.cols, sizeof(*net->active_prev));

    for(u32 col = 0; col < p.cols; ++col) {
        net->active[col] = 0;
        net->predicted[col] = 0;

        net->active_prev[col] = 0;
    }

    init_l6_segment_tensor(net, f_p);
}

void location_layer_predict(location_layer_t* net, feature_layer_t* f_net) {
    segment_t* location_context_pointer = net->in_segments.location_context;
    segment_t* feature_context_pointer = net->in_segments.feature_context;

    for(u32 col = 0; col < net->p.cols; ++col) {
        u32 pred_bitarray = 0; // 000...000
        for(u32 cell = 0; cell < net->p.cells; ++cell) {
            u32 num_spiking_segments = 0;

            htm_prediction_integrate_context(&num_spiking_segments,
                location_context_pointer, net->active, 
                LOCATION_INDEX_TYPE, net->p.location_segments, 
                net->p.htm.permanence_threshold, net->p.htm.segment_spiking_threshold);
            
            htm_prediction_integrate_context(&num_spiking_segments,
                feature_context_pointer, f_net->active, 
                FEATURE_INDEX_TYPE, net->p.feature_segments, 
                net->p.htm.permanence_threshold, net->p.htm.segment_spiking_threshold);

            // cell is predicted if there is at least one spiking segment
            u32 cell_is_predicted = num_spiking_segments >= 1;
            pred_bitarray |= (cell_is_predicted << cell);

            location_context_pointer += net->p.location_segments;
            feature_context_pointer += net->p.feature_segments;
        }
        
        net->predicted[col] = pred_bitarray;
    }
}

void location_layer_activate(location_layer_t* net, vec2d movement, feature_layer_t* f_net) {
    for(u32 it = 0; it < net->p.cols; ++it) {
        net->active_prev[it] = net->active[it]; // remember the active cols here
        net->active[it] = 0; // set all columns to be inactive, active columns will be selectively activated later in this function
    }

    // 1. movement -> active columns i.e. shift the grid cell activity with movement
    for(u32 col_it = 0; col_it < net->p.cols; ++col_it) {
        // if the column was active, we need to shift this activity through movement, and compute the next column that should be active
        if(net->active_prev[col_it]) {
            u32 cols_sqrt = 1 << net->p.log_cols_sqrt; 

            u32 x = col_it % cols_sqrt;
            u32 y = col_it >> net->p.log_cols_sqrt; // col_it / cols_sqrt

            u32 new_x = (x + (movement.x >> net->p.log_scale.x)) % cols_sqrt; // grid cells wrap around
            u32 new_y = (y + (movement.y >> net->p.log_scale.y)) % cols_sqrt; // grid cells wrap around

            u32 col_index = new_x + (new_y << net->p.log_cols_sqrt);

            net->active[col_index] = 1;
        }
    }

    // 2. predicted cells & active columns -> active cells
    htm_activate(net->active, net->active, net->predicted, net->p.cols, net->p.cells);

    // 3. learning (involves traversing the segments data)
    segment_t* location_context_pointer = net->in_segments.location_context;
    segment_t* feature_context_pointer = net->in_segments.feature_context;

    for(u32 col = 0; col < net->p.cols; ++col) {
        u32 pred_bitarray = net->predicted[col];

        /**
         * If the columns is actived but no cell was predicted, we have to select the "winning cell"
         * to that end, we iterate through all the cells and we find the cell with the segment that was closest to becoming active
         */
        u32 winning_cell = 0;
        u32 winning_connection_count = 0;
        if(net->active[col] != 0 && pred_bitarray == 0) {
            // We find the cell with the segment that had most activity
            htm_learning_pick_winner_cell(
                &winning_cell, &winning_connection_count, 
                location_context_pointer, net->p.location_segments, net->p.cells
            );

            htm_learning_pick_winner_cell(
                &winning_cell, &winning_connection_count, 
                feature_context_pointer, net->p.feature_segments, net->p.cells
            );
        }

        for(u32 cell = 0; cell < net->p.cells; ++cell) {
            htm_learning_adjust_permanences(
                net->active_prev, location_context_pointer,
                winning_cell, winning_connection_count,
                net->active[col], net->predicted[col],
                LOCATION_INDEX_TYPE, net->p.location_segments,
                net->p.htm, cell
            );

            htm_learning_adjust_permanences(
                f_net->active_prev, feature_context_pointer,
                winning_cell, winning_connection_count,
                net->active[col], net->predicted[col],
                FEATURE_INDEX_TYPE, net->p.feature_segments,
                net->p.htm, cell
            );

            location_context_pointer += net->p.location_segments;
            location_context_pointer += net->p.feature_segments;
        }  
    }
}
