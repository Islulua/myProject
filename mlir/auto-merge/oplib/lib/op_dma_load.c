#define MAX_SHAPE_DIM 6

typedef enum LOAD_MODE {
    LOAD_NORMAL = 0,
    LOAD_CNORM2Cx = 1,
    LOAD_TENSOR = 2,
    LOAD_GEMM_BATCH = 3,
    LOAD_MODE_END
} LOAD_MODE;

typedef struct WINDOW_PARAM {
    uint16_t Ky;
    uint16_t Kx;
    uint16_t Kt;
    uint16_t Sy;
    uint16_t Sx;
    uint16_t St;
    uint16_t up_pad_y;
    uint16_t up_pad_x;
    uint16_t up_pad_t;
    uint16_t down_pad_y;
    uint16_t down_pad_x;
    uint16_t down_pad_t;
} WINDOW_PARAM;

typedef struct SPATIAL_PARAM {
    int32_t spatial_sharding[MAX_SHAPE_DIM];
    int32_t temporal_slice[MAX_SHAPE_DIM];
    int32_t dp_dim_x;
    int32_t dp_dim_y;
    int32_t tp_dim_x;
    int32_t tp_dim_y;
    int32_t layout;
    int32_t first_dim;
    uint8_t parallel; // 0: DP, 1: TP, 2: DP+TP
    uint8_t dp_inner; // Valid when MIX, 0: tp is inner, 1: dp is inner
    uint8_t if_hwt_split; // if height, width, or time is split
    LOAD_MODE load_type;
    WINDOW_PARAM window;
} SPATIAL_PARAM;

typedef struct LOADVAR_PARAM {
    uint64_t addr;
    int32_t tile_id_this;
    int32_t temporal_slice[MAX_SHAPE_DIM];

    int32_t dma_mode;
    int32_t base_offset;
    SPATIAL_PARAM spatial_param;
} LOADVAR_PARAM;

typedef enum DATA_FORMAT {
    Fmt_INT8 = 0,
    Fmt_INT16 = 1,
    Fmt_INT32 = 2,
    Fmt_INT64 = 3,
    Fmt_FP16 = 4,
    Fmt_FP32 = 5,
    Fmt_END
} DATA_FORMAT;

struct L_SHAPE {
    int32_t dim;
    int32_t shape_whole[MAX_SHAPE_DIM];
    int32_t shape_start[MAX_SHAPE_DIM];
    int32_t shape_slice[MAX_SHAPE_DIM];
    int32_t shape_real[MAX_SHAPE_DIM];
};

typedef struct G_SHAPE {
    int32_t spatial_start[MAX_SHAPE_DIM];
    int32_t spatial_end[MAX_SHAPE_DIM];
    int32_t dynamic_offset[MAX_SHAPE_DIM];
    int32_t shape[MAX_SHAPE_DIM];
    int32_t dim;
    int32_t done;
    int32_t batch_offset[MAX_SHAPE_DIM];
} G_SHAPE;


typedef struct TSR {
    uint64_t addr;
    DATA_Format format;
    L_SHAPE* shape;
} TSR;

static void process_first_time(LOADVAR_PARAM *param, G_SHAPE* in_shape, TSR* out) {
    LAYOUT_MODE layout = param->spatial_param.layout;
    LOAD_MODE load_mode = param->spatial_param.load_type;
    bool consider_window = param->spatial_param.if_hwt_split;
    int32_t dim = (in_shape->dim == 0) ? 1 : in_shape->dim;
    int32_t *spatial_start = in_shape->spatial_start;
    int32_t *spatial_end = in_shape->spatial_end;
    int32_t *dynamic_offset = in_shape->dynamic_offset;
    int32_t *shape_whole = in_shape->shape;
    int32_t total_tile_num = param->spatial_param.dp_dim_x * param->spatial_param.dp_dim_y * param->spatial_param.tp_dim_x * param->spatial_param.tp_dim_y;

    int32_t spatial_slice;
    int32_t shard_mul = 1;
    bool is_split_zero = false;

    for (int32_t i = 0; i < dim; i++) {
        int32_t split = param->spatial_param.spatial_sharding[i];
        shard_mul *= (split > 0 ? split : 1);
        is_split_zero = (split > 0 ? false : true);
    }
    uint32_t tile_id = param->tile_id_this;
    int32_t total_split = 0;
    for (int32_t dim_idx = dim - 1; dim_idx >= 0; dim_idx--) {
        int32_t split = param->spatial_param.spatial_sharding[dim_idx];
        total_split += split;
        split = split > 0 ? split : 1;
        int32_t cur_split;

        if (load_type >= LOAD_TENSOR) {
            cur_split = tile_id % split;
            tile_id = tile_id / split;
            if (dim_idx == 0) {
                tile_id = tile_id / (param->spatial_param.dp_dim_x * param->spatial_param.dp_dim_y);
            }
        } else {
            if (param.spatial_param.dp_inner) {
                cur_split = tile_id % split;
                tile_id = tile_id / split;
                if (dim_idx == 0 && (param->spatial_param.spatial_sharding[dim - 1] == 1)) {
                    tile_id = tile_id / (param->spatial_param.tp_dim_x * param->spatial_param.tp_dim_y);
                }
            } else {
                if (dim_idx == (dim - 1)) {
                    tile_id = tile_id / (param->spatial_param.tp_dim_x * param->spatial_param.tp_dim_y);
                }
                tile_id = ((total_tile_num != shard_mul) && (param->spatial_param.parallel == 0)) ? (tile_id % shard_mul) : tile_id;
                cur_split = tile_id % split;
                tile_id = tile_id / split;
            }
        }

        spatial_slice = CEIL(shape_whole[dim_idx], split);
        bool need_align = ((is_cx_layout(layout) != ALIGN_NOT) || (load_type != LOAD_NORMAL));
        if (spatial_slice != 0) {
            spatial_slice = 
                  (need_align && ((dim_idx == dim - 1) || (dim_idx == dim - 2))) ?
                        ALIGN_FUNC(spatial_slice, get_cx_align_base(spatial_slice, out->format)) :
                        spatial_slice;
        }
        spatial_slice = MAX(spatial_slice, param->spatial_param.temporal_slice[dim_idx]);

        if (consider_window) {
            // 根据k s pad 算出output的相关start len， 根据output反推input起始位置
        } else {
            // gemm的weight不更新batch的offset
            if (!((load_type == LOAD_GEMM_BATCH) && (dim >= 3 + dim_idx))) {
                spatial_start[dim_idx] = cur_split * spatial_slice;
                spatial_end[dim_idx] = spatial_start[dim_idx] + spatial_slice - 1;
            }
        }

        if (!((load_type == LOAD_GEMM_BATCH) && (dim >= 3 + dim_idx))) {
            dynamic_offset[dim_idx] = spatial_start[dim_idx];
        }

        // 没有被分到数据的tile
        if ((tile_id >= 1 && !is_split_zero)) {
            for (int32_t i = 0; i < dim; i++) {
                spatial_start[i] = (DYNAMIC_SHAPE_TEST ? 0 : in_shape->shape[i]);
                spatial_end[i] = (DYNAMIC_SHAPE_TEST ? 0 : in_shape->shape[i]) - 1;
                dynamic_offset[i] = spatial_start[i];
            }
        }
    }
}

uint64_t op_dma_load(LOADVAR_PARAM *param, G_SHAPE* in_shape, TSR* out, uint8_t first_time) {
    LAYOUT_MODE layout_mode = param->spatial_param.layout;
    LOAD_MODE load_mode = param->spatial_param.load_type;
    int32_t dim = (in_shape->dim == 0) ? 1 : in_shape->dim;
    if (in_shape->dim == 0) {
        in_shape->shape[0] = 1;
    }

    bool consider_window = param->spatial_param.if_hwt_split;

    if (first_time) {
        process_first_time(param, in_shape, out);
    }
    // TODO: 进行dma数据load
}