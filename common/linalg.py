import torch


def project_point_to_plane(points, normal_vec, offset):

    projected_proints = points - normal_vec * \
        (points @ normal_vec.t() + offset / (normal_vec @ normal_vec.T))
    return projected_proints


def traverse_along_normal_vec(projected_points, normal_vector, traverse_steps, direction_vector=None):
    projected_points = projected_points.unsqueeze(1)
    if direction_vector is None:
        unit_normal_vec = normal_vector / torch.norm(normal_vector)
        unit_normal_vec = unit_normal_vec.unsqueeze(1)
        traverse_points = projected_points + unit_normal_vec * traverse_steps
    else:
        unit_direction_vec = direction_vector / torch.norm(direction_vector)
        traverse_points = projected_points + unit_direction_vec / \
            (direction_vector @ normal_vector.t()) * traverse_steps
    return traverse_points
