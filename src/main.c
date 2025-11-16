#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <unistd.h> // For sleep function

#include "grid_environment.h"
#include "sensor_module.h"
#include "learning_module.h"
#include "motor_policy.h"

int main(int argc, char *argv[]) {  

// Put the error message in a char array:
    const char error_message[] = "Error: usage: %s X\n\t \
        0 is for training from scratch\n";


    /* Error Checking */
    if(argc < 0) {
        printf(error_message, argv[0]);
        exit(1);
    }

    u32 num_step = 10;

    grid_t env;
    u32 env_sidelen = 10;
    init_grid_env(&env, env_sidelen, env_sidelen);
    populate_grid_env_random(&env);

    grid_t patch; // grid_t is an ugly abstraction for the patch
    u32 patch_sidelen = 3;
    init_grid_env(&patch, patch_sidelen, patch_sidelen);
    uvec2d patch_center = (uvec2d) { .x = patch_sidelen / 2, .y = patch_sidelen / 2 };

    bounds_t bounds = get_bounds(env_sidelen, env_sidelen, patch_sidelen, patch_sidelen);

    uvec2d agent_location = { .x = 5, .y = 1 }; // start location

    u32 num_cols = 1024;

    grid_sm sm;
    init_sensor_module(&sm, GRID_ENV_MIN_VALUE, GRID_ENV_MAX_VALUE, num_cols);

    print_pooler(&sm.pooler);

    random_motor_policy_t motor_policy;
    init_random_motor_policy(&motor_policy, agent_location, bounds, num_step);

    features_t f;
    init_features(&f, sm.pooler.params.num_minicols, sm.pooler.params.top_k);

    htm_params_t htm_params = (htm_params_t) {
        .permanence_threshold = REPR_u8(0.5),
        .segment_spiking_threshold = 15,

        .perm_increment = REPR_u8(0.06f),
        .perm_decrement = REPR_u8(0.04f),
        .perm_decay = 1 // 1/256, the smallest possible non-zero decay
    };

    extended_htm_params_t ext_htm_params = (extended_htm_params_t) {
        .feedforward_permanence_threshold = REPR_u8(0.5),
        .context_permanence_threshold = REPR_u8(0.5),

        .feedforward_activation_threshold = 3,
        .context_activation_threshold = 18,

        .min_active_cells = 10,
    };
 
    learning_module lm;
    init_learning_module(
        &lm, 
        (output_layer_params_t) {
            .cells = 1024, // cells per col

            .internal_context_segments = 10,
            .external_context_segments = 0,
            .context_segments = 10,
            
            .htm = htm_params,
            .extended_htm = ext_htm_params
        }, 
        (feature_layer_params_t) {
            .cols = num_cols,
            .cells = 10, // cells per col

            .feature_segments = 5,
            .location_segments = 5,

            .htm = htm_params
        },
        (location_layer_params_t) {
            .cols = 1024,
            .log_cols_sqrt = (u32) log2(sqrt(1024)), // 5
            .cells = 10,

            .location_segments = 5,
            .feature_segments = 5,

            .log_scale = (uvec2d) { .x = 0, .y = 0 },

            .htm = htm_params
        }
    );

    vec2d movement = { .x = 0, .y = 0 };

    print_grid(&env);

    for(u32 step = 0; step < num_step; ++step) {
        printf("--- step %u: agent at location (%u, %u)\n", step, agent_location.x, agent_location.y);
        extract_patch(&patch, &env, agent_location, patch_sidelen);

        print_grid(&patch);

        sensor_module(sm, &f, patch, patch_center);

        printf("column activation sparsity: ");
        print_spvec_u8(f.active_columns, f.num_columns);

        learning_module_step(&lm, f, movement);

        print_packed_spvec_u32(lm.output_net.active, lm.output_net.p.cells);

        movement = random_motor_policy(&motor_policy, f);

        agent_location.x += movement.x;
        agent_location.y += movement.y;

        printf("\n");
    }

    return 0;
}
