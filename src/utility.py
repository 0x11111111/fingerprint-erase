import math
from types import SimpleNamespace


def sn2dict(sn):
    d = dict()
    if isinstance(sn, SimpleNamespace):
        for k, v in sn.__dict__.items():
            if isinstance(v, SimpleNamespace):
                d[k] = sn2dict(v)
            else:
                d[k] = v

    return d


def calculate_distance_sn(coord_sn_1, coord_sn_2):
    return math.sqrt((coord_sn_1.x - coord_sn_2.x) ** 2 + (coord_sn_1.y - coord_sn_2.y) ** 2)


def calculate_distance_array(coord_array_1, coord_array_2):
    return math.sqrt((coord_array_1[0] - coord_array_2[0]) ** 2 + (coord_array_1[1] - coord_array_2[1]) ** 2)


def intersect_line_circle(circle_center, circle_radius, pt1, pt2, full_line=False, tangent_tol=1e-4):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    param circle_center: The (x, y) location of the circle center
    param circle_radius: The radius of the circle
    param pt1: The (x, y) location of the first point of the segment
    param pt2: The (x, y) location of the second point of the segment
    param full_line: True to find intersections along full line - not just in the segment.  False will just
        return intersections within the segment.
    param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to
        consider it a tangent
    return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at
        which the circle intercepts a line segment.

    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2) ** 0.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant ** 0.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant ** 0.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))
        ]  # This makes sure the order along the segment is correct
        if not full_line:
            # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in
                                      intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(
                discriminant) <= tangent_tol:
            # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections