"""
author: William Andersson
"""

from configs import (
    configs_Ball2D_BM,
    configs_BM_weights,
    configs_RBM,
    configs_Rectangle,
)
from configs.dataset_configs import (
    DATA_DICTS,
    RBM_pen_func,
    ball2D_pen_func,
    rect_pen_func,
    simplex_pen_func,
)

# RBM
(
    param_list_RBM_STANDARD_NJODE,
    overview_dict_RBM_STANDARD_NJODE,
    plot_paths_RBM_STANDARD_NJODE,
) = configs_RBM.get_RBM_STANDARD_NJODE_config()
(
    param_list_RBM_STANDARD_OPTIMAL_PROJ,
    overview_dict_RBM_STANDARD_OPTIMAL_PROJ,
    plot_paths_RBM_STANDARD_OPTIMAL_PROJ,
) = configs_RBM.get_RBM_STANDARD_OPTIMAL_PROJ_config()
(
    param_list_RBM_MORE_BOUNCES_NJODE,
    overview_dict_RBM_MORE_BOUNCES_NJODE,
    plot_paths_RBM_MORE_BOUNCES_NJODE,
) = configs_RBM.get_RBM_MORE_BOUNCES_NJODE_config()
(
    param_list_RBM_MORE_BOUNCES_OPTIMAL_PROJ,
    overview_dict_RBM_MORE_BOUNCES_OPTIMAL_PROJ,
    plot_paths_RBM_MORE_BOUNCES_OPTIMAL_PROJ,
) = configs_RBM.get_RBM_MORE_BOUNCES_OPTIMAL_PROJ_config()


# Rectangle
(
    param_list_RECTANGLE_STANDARD_NJODE,
    overview_dict_RECTANGLE_STANDARD_NJODE,
    plot_paths_RECTANGLE_STANDARD_NJODE_dict,
) = configs_Rectangle.get_RECTANGLE_STANDARD_NJODE_config()
(
    param_list_RECTANGLE_STANDARD_OPTIMAL_PROJ,
    overview_dict_RECTANGLE_STANDARD_OPTIMAL_PROJ,
    plot_paths_RECTANGLE_STANDARD_OPTIMAL_PROJ_dict,
) = configs_Rectangle.get_RECTANGLE_STANDARD_OPTIMAL_PROJ_config()
(
    param_list_RECTANGLE_STANDARD_VERTEX_APPROACH,
    overview_dict_RECTANGLE_STANDARD_VERTEX_APPROACH,
    plot_paths_RECTANGLE_STANDARD_VERTEX_APPROACH_dict,
) = configs_Rectangle.get_RECTANGLE_STANDARD_VERTEX_APPROACH_config()
(
    param_list_RECTANGLE_WIDER_WITH_MU_NJODE,
    overview_dict_RECTANGLE_WIDER_WITH_MU_NJODE,
    plot_paths_RECTANGLE_WIDER_WITH_MU_NJODE_dict,
) = configs_Rectangle.get_RECTANGLE_WIDER_WITH_MU_NJODE_config()
(
    param_list_RECTANGLE_WIDER_WITH_MU_OPTIMAL_PROJ,
    overview_dict_RECTANGLE_WIDER_WITH_MU_OPTIMAL_PROJ,
    plot_paths_RECTANGLE_WIDER_WITH_MU_OPTIMAL_PROJ_dict,
) = configs_Rectangle.get_RECTANGLE_WIDER_WITH_MU_OPTIMAL_PROJ_config()
(
    param_list_RECTANGLE_WIDER_WITH_MU_VERTEX_APPROACH,
    overview_dict_RECTANGLE_WIDER_WITH_MU_VERTEX_APPROACH,
    plot_paths_RECTANGLE_WIDER_WITH_MU_VERTEX_APPROACH_dict,
) = configs_Rectangle.get_RECTANGLE_WIDER_WITH_MU_VERTEX_APPROACH_config()


# BM weights
(
    param_list_BM_WEIGHTS_RECTANGLE_STANDARD_NJODE,
    overview_dict_BM_WEIGHTS_RECTANGLE_STANDARD_NJODE,
    plot_paths_BM_WEIGHTS_RECTANGLE_STANDARD_NJODE_dict,
) = configs_BM_weights.get_BM_WEIGHTS_RECTANGLE_STANDARD_NJODE_config()
(
    param_list_BM_WEIGHTS_RECTANGLE_STANDARD_OPTIMAL_PROJ,
    overview_dict_BM_WEIGHTS_RECTANGLE_STANDARD_OPTIMAL_PROJ,
    plot_paths_BM_WEIGHTS_RECTANGLE_STANDARD_OPTIMAL_PROJ_dict,
) = configs_BM_weights.get_BM_WEIGHTS_RECTANGLE_STANDARD_OPTIMAL_PROJ_config()
(
    param_list_BM_WEIGHTS_RECTANGLE_STANDARD_VERTEX_APPROACH,
    overview_dict_BM_WEIGHTS_RECTANGLE_STANDARD_VERTEX_APPROACH,
    plot_paths_BM_WEIGHTS_RECTANGLE_STANDARD_VERTEX_APPROACH_dict,
) = configs_BM_weights.get_BM_WEIGHTS_RECTANGLE_STANDARD_VERTEX_APPROACH_config()
(
    param_list_BM_WEIGHTS_SIMPLEX2D_NJODE,
    overview_dict_BM_WEIGHTS_SIMPLEX2D_NJODE,
    plot_paths_BM_WEIGHTS_SIMPLEX2D_NJODE_dict,
) = configs_BM_weights.get_BM_WEIGHTS_SIMPLEX2D_NJODE_config()
(
    param_list_BM_WEIGHTS_SIMPLEX2D_OPTIMAL_PROJ,
    overview_dict_BM_WEIGHTS_SIMPLEX2D_OPTIMAL_PROJ,
    plot_paths_BM_WEIGHTS_SIMPLEX2D_OPTIMAL_PROJ_dict,
) = configs_BM_weights.get_BM_WEIGHTS_SIMPLEX2D_OPTIMAL_PROJ_config()
(
    param_list_BM_WEIGHTS_SIMPLEX2D_VERTEX_APPROACH,
    overview_dict_BM_WEIGHTS_SIMPLEX2D_VERTEX_APPROACH,
    plot_paths_BM_WEIGHTS_SIMPLEX2D_VERTEX_APPROACH_dict,
) = configs_BM_weights.get_BM_WEIGHTS_SIMPLEX2D_VERTEX_APPROACH_config()
(
    param_list_BM_WEIGHTS_SIMPLEX3D_NJODE,
    overview_dict_BM_WEIGHTS_SIMPLEX3D_NJODE,
    plot_paths_BM_WEIGHTS_SIMPLEX3D_NJODE_dict,
) = configs_BM_weights.get_BM_WEIGHTS_SIMPLEX3D_NJODE_config()
(
    param_list_BM_WEIGHTS_SIMPLEX3D_OPTIMAL_PROJ,
    overview_dict_BM_WEIGHTS_SIMPLEX3D_OPTIMAL_PROJ,
    plot_paths_BM_WEIGHTS_SIMPLEX3D_OPTIMAL_PROJ_dict,
) = configs_BM_weights.get_BM_WEIGHTS_SIMPLEX3D_OPTIMAL_PROJ_config()
(
    param_list_BM_WEIGHTS_SIMPLEX3D_VERTEX_APPROACH,
    overview_dict_BM_WEIGHTS_SIMPLEX3D_VERTEX_APPROACH,
    plot_paths_BM_WEIGHTS_SIMPLEX3D_VERTEX_APPROACH_dict,
) = configs_BM_weights.get_BM_WEIGHTS_SIMPLEX3D_VERTEX_APPROACH_config()


# Ball2D
(
    param_list_BALL2D_STANDARD_NJODE,
    overview_dict_BALL2D_STANDARD_NJODE,
    plot_paths_BALL2D_STANDARD_NJODE_dict,
) = configs_Ball2D_BM.get_BALL2D_STANDARD_NJODE_config()
(
    param_list_BALL2D_STANDARD_OPTIMAL_PROJ,
    overview_dict_BALL2D_STANDARD_OPTIMAL_PROJ,
    plot_paths_BALL2D_STANDARD_OPTIMAL_PROJ_dict,
) = configs_Ball2D_BM.get_BALL2D_STANDARD_OPTIMAL_PROJ_config()
(
    param_list_BALL2D_LARGE_NJODE,
    overview_dict_BALL2D_LARGE_NJODE,
    plot_paths_BALL2D_LARGE_NJODE_dict,
) = configs_Ball2D_BM.get_BALL2D_LARGE_NJODE_config()
(
    param_list_BALL2D_LARGE_OPTIMAL_PROJ,
    overview_dict_BALL2D_LARGE_OPTIMAL_PROJ,
    plot_paths_BALL2D_LARGE_OPTIMAL_PROJ_dict,
) = configs_Ball2D_BM.get_BALL2D_LARGE_OPTIMAL_PROJ_config()
(
    param_list_TINY_NJODE,
    overview_dict_TINY_NJODE,
    plot_paths_TINY_NJODE_dict,
) = configs_Ball2D_BM.get_TINY_NJODE_config()
(
    param_list_TINY_OPTIMAL_PROJ,
    overview_dict_TINY_OPTIMAL_PROJ,
    plot_paths_TINY_OPTIMAL_PROJ_dict,
) = configs_Ball2D_BM.get_TINY_OPTIMAL_PROJ_config()


CONVEX_PEN_FUNCS = {
    "RBM_STANDARD": RBM_pen_func(DATA_DICTS["RBM_STANDARD"]),
    "RBM_MORE_BOUNCES": RBM_pen_func(DATA_DICTS["RBM_MORE_BOUNCES"]),
    "RECTANGLE_STANDARD": lambda Y: rect_pen_func(Y, "RECTANGLE_STANDARD"),
    "RECTANGLE_WIDER_WITH_MU": lambda Y: rect_pen_func(Y, "RECTANGLE_WIDER_WITH_MU"),
    "BM_WEIGHTS_RECTANGLE_STANDARD": lambda Y: rect_pen_func(Y, "RECTANGLE_STANDARD"),
    "BM_WEIGHTS_SIMPLEX2D": simplex_pen_func,
    "BM_WEIGHTS_SIMPLEX3D": simplex_pen_func,
    "BALL2D_STANDARD": ball2D_pen_func(DATA_DICTS["BALL2D_STANDARD"]),
    "BALL2D_LARGE": ball2D_pen_func(DATA_DICTS["BALL2D_LARGE"]),
    "BALL2D_TINY": ball2D_pen_func(DATA_DICTS["BALL2D_TINY"]),
}
