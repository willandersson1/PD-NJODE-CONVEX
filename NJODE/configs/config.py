"""
author: William Andersson
"""

from configs import (
    configs_Ball2D_BM,
    configs_BM_weights,
    configs_FBM,
    configs_RBM,
    configs_Rectangle,
)
from configs.dataset_configs import (
    DATA_DICTS,
    rect_pen_func,
    standard_2_norm_for_lb_ub,
    zero_pen_func,
)

# TODO use RNN with no residuals, as discussed

# FBM
param_list_FBM1, overview_dict_FBM1, plot_paths_FBM_dict = configs_FBM.get_FBM1_config()

# RBM
param_list_RBM, overview_dict_RBM, plot_paths_RBM_dict = configs_RBM.get_RBM_config()


# Rectangle
param_list_Rectangle, overview_dict_Rectangle, plot_paths_Rectangle_dict = (
    configs_Rectangle.get_Rectangle_config()
)

# Rectangle vertex approach
(
    param_list_Rectangle_vertex_approach,
    overview_dict_Rectangle_vertex_approach,
    plot_paths_Rectangle_vertex_approach_dict,
) = configs_Rectangle.get_Rectangle_vertex_approach_config()

# Triangle BM weights vertex approach
(
    param_list_Triangle_BM_weights,
    overview_dict_Triangle_BM_weights,
    plot_paths_Triangle_BM_weights_dict,
) = configs_BM_weights.get_Triangle_BM_weights_config()

# Ball2D with NJODE
param_list_Ball2D_BM, overview_dict_Ball2D_BM, plot_paths_Ball2D_BM_dict = (
    configs_Ball2D_BM.get_Ball2D_BM_config()
)

CONVEX_PEN_FUNCS = {
    "RBM_1_dict": lambda Y: standard_2_norm_for_lb_ub(
        Y,
        DATA_DICTS["RBM_1_dict"]["lb"],
        DATA_DICTS["RBM_1_dict"]["ub"],
    ),
    "Rectangle_1_dict": lambda Y: rect_pen_func(Y, DATA_DICTS["Rectangle_1_dict"]),
    "Triangle_BM_weights_1_dict": zero_pen_func,
}
