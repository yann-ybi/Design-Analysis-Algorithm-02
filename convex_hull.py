import math
import sys
from typing import List
from typing import Tuple

EPSILON = sys.float_info.epsilon
Point = Tuple[int, int]

# for every list of point the lowest point(s) with the hightest y(s) are part of the convex hull
# for every 2 adjacent hull points, there is a hull segment
# for every hull segment the point from the list of points forming the wider angle it is part of the convex hull
# for every points of a convex hull, there is only one path from a hull point back to itself passing by all convex hull points
# the convex hull algorithm reaches termination when that circular path is made.
def base_case_hull(points: List[Point]) -> List[Point]:
    """ 
    Base case of the recursive algorithm.
    This base case compute a convex hull given a list of points
    """
    if len(points) < 4:    
        return points

    fp_and_idx = first_point(points)
    new_points = [fp_and_idx[0]]

    first_segL(new_points, points)
    first_segR(new_points, points)

    hull_point1 = new_points[1]
    hull_point2 = new_points[2]
    
    points.append(new_points[0])
    hull_seg(hull_point1, hull_point2, points, new_points)

    while new_points[len(new_points) - 1] != new_points[0]:
        hull_seg(new_points[ len(new_points) - 2 ], new_points[ len(new_points) - 1 ], points, new_points)

    new_points.pop(0)

    return new_points


# for every list of points with a certain length there is a split in half based on the points x axis to perform
# for every splited list of points small enough there are 2 convex hulls to complute using the base case
# for every splited list convex hull computed there is a merge to perform
# for every merge there is one convex hull made from two distinct ones
# for every merge the area of the merged convex hull is at least as much as the area of the child hull combined
# for every merge we go up on our binary tree by 1 untill we reach the root
sorted = False
def compute_hull(points: List[Point]) -> List[Point]:
    """
    Given a list of points.
    Calls the base case for if the number of points is less than 7.
    return a list of convex hulls
    else split the list of points into 2 lists.
        recursively call compute_hull on each list
        recursively call on the return values of the compute hull call
        return a list of merged convex hulls
    """
    if len(points) <= 6:
        return base_case_hull(points)
    else:  
        global sorted
        if not sorted:
            points.sort()
            sorted = True
        
        mid = int(len(points) / 2)
        try:
            merged = merge_hull(compute_hull(points[:mid]), compute_hull(points[mid:]))
        except:
            merged = base_case_hull(points)

        return merged

# for every pair of convex hull there is only one convex hull possible from a merge
# for every merge 2 new convex hull segments needs to be formed
    # for every merge segment formed, there are 2 points, one for each child hull to merge
# for each merged hull there is one circular path with all none other points inside
def merge_hull(left_half: List[Point], right_half: List[Point]) -> List[Point]:
    """
    merges 2 hulls into 1.
    """
    clockwise_sort(left_half)
    clockwise_sort(right_half)
    hulls = halfs_hull_pts(left_half, right_half)
    tophull = hulls[0]
    bothull = hulls[1]

    left_top = tophull[0]
    right_bot = bothull[1]
    
    result = []
    start_idx = bothull[2]
    end_idx = tophull[2]

    curr_idx = start_idx
    while curr_idx % len(left_half) != end_idx:
        result.append(left_half[curr_idx % len(left_half)])
        curr_idx += 1

    result.append(left_top)
    
    start_idx = tophull[3]
    end_idx = bothull[3]

    curr_idx = start_idx
    while curr_idx % len(right_half) != end_idx:
        result.append(right_half[curr_idx % len(right_half)])
        curr_idx += 1

    result.append(right_bot)
    return result

def y_intercept(p1: Point, p2: Point, x: int) -> float:
    """
    Given two points, p1 and p2, an x coordinate from a vertical line,
    compute and return the the y-intercept of the line segment p1->p2
    with the vertical line passing through x.
    """
    x1, y1 = p1
    x2, y2 = p2
    slope = (y2 - y1) / (x2 - x1)
    return y1 + (x - x1) * slope


def triangle_area(a: Point, b: Point, c: Point) -> float:
    """
    Given three points a,b,c,
    computes and returns the area defined by the triangle a,b,c.
    Note that this area will be negative if a,b,c represents a clockwise sequence,
    positive if it is counter-clockwise,
    and zero if the points are collinear.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    return ((cx - bx) * (by - ay) - (bx - ax) * (cy - by)) / 2


def is_clockwise(a: Point, b: Point, c: Point) -> bool:
    """
    Given three points a,b,c,
    returns True if and only if a,b,c represents a clockwise sequence
    (subject to floating-point precision)
    """
    return triangle_area(a, b, c) < -EPSILON


def is_counter_clockwise(a: Point, b: Point, c: Point) -> bool:
    """
    Given three points a,b,c,
    returns True if and only if a,b,c represents a counter-clockwise sequence
    (subject to floating-point precision)
    """
    return triangle_area(a, b, c) > EPSILON


def collinear(a: Point, b: Point, c: Point) -> bool:
    """
    Given three points a,b,c,
    returns True if and only if a,b,c are collinear
    (subject to floating-point precision)
    """
    return abs(triangle_area(a, b, c)) <= EPSILON


def clockwise_sort(points: List[Point]):
    """
    Given a list of points, sorts those points in clockwise order about their centroid.
    Note: this function modifies its argument.
    """
    x_mean = sum(p[0] for p in points) / len(points)
    y_mean = sum(p[1] for p in points) / len(points)

    def angle(point: Point):
        return (math.atan2(point[1] - y_mean, point[0] - x_mean) + 2 * math.pi) % (2 * math.pi)

    points.sort(key=angle)
    return


def first_point(points: List[Point]) -> List[Point]:
    """
    Given a list of points, pop the point with the highest y and push it into a new list.
    Note: this function modifies its argument.
    """
    first_point_idx = highes_y_index(points)
    first_point = points[first_point_idx]
    points.pop(first_point_idx)

    return [first_point, first_point_idx]
    
def hull_segL(base_point: Point, hull_point: Point, points: List[Point], new_points = List[Point] ) -> List[Point]:
    """
    Given a list of points, the hull point with the higest y, a base point forming a segment perpendicular to the y axis with a x = 0 and a list of hull points.
    find the left adjacent hull point, remove that point from list of points, add it to a hull point list and return it
    Note: this function modifies its argument.
    """
    hull_point2_idx = 0
    degL = angleL(base_point, hull_point, points[hull_point2_idx])
    for i, point in enumerate(points):
        new_degL = angleL(base_point, hull_point, point)
        if new_degL < degL:
            hull_point2_idx = i
            degL = new_degL

    new_points.append(points[hull_point2_idx])
    points.pop(hull_point2_idx)

    return

def hull_segR(base_point: Point, hull_point: Point, points: List[Point], new_points = List[Point] ) -> List[Point]:
    """
    Given a list of points, the hull point with the higest y, a base point on the right side forming a segment perpendicular to the y axis with and a list of hull points.
    find the right adjacent hull point, remove that point from list of points, add it to a hull point list and return it
    Note: this function modifies its argument.
    """

    hull_point2_idx = 0
    degR = angleR(base_point, hull_point, points[hull_point2_idx])
    for i, point in reversed(list(enumerate(points))):
        new_degR = angleR(base_point, hull_point, point)
        if new_degR < degR:
            hull_point2_idx = i
            degR = new_degR

    new_points.insert(0, points[hull_point2_idx])
    points.pop(hull_point2_idx)

    return


def first_segL(new_points: List[Point], points: List[Point]) -> List[Point]:
    """
    Given a list of hull points, a list of points.
    find a base point on the left side perpendicular to the x axis by calling base_pointL
    return the list of Points returned by hull_segL
    """
    base_point = base_pointL(new_points[0])
    return hull_segL( base_point, new_points[0], points, new_points)

def first_segR(new_points: List[Point], points: List[Point]) -> List[Point]:
    """
    Given a list of hull points, a list of points.
    find a base point on the right side perpendicular to the x axis by calling base_pointR
    return the list of Points returned by hull_segR
    """
    base_point = base_pointR(new_points[0])
    return hull_segR( base_point, new_points[0], points, new_points)


def base_pointR(point: Point) -> List[Point]:
    """
    Given a max x value and a point.
    find a base point on the right side perpendicular to the y axis
    return that base point
    """
    base_point = [point[0], point[1]]
    return base_point

def base_pointL(new_point: Point) -> List[Point]:
    """
    Given a point.
    find a base point on the left side perpendicular to the y axis
    return that base point
    """
    base_point = [0, new_point[1]]
    return base_point

def angleL(p1: Point, p2: Point, p3: Point):
    """
    Given three points.
    Find the left angle from p1, p2 seg to p2, p3 seg and return it
    """
    deg = math.degrees(math.atan2(p1[1]-p2[1], p1[0]-p2[0]) - math.atan2(p3[1]-p2[1], p3[0]-p2[0]))

    return  360 - (deg + 360 if deg < 0 else deg)

def angleR(p1: Point, p2: Point, p3: Point):
    """
    Given three points.
    Find the right angle from p2, p3 seg to p1, p2 seg and return it
    """
    deg = math.degrees(math.atan2(p3[1]-p2[1], p3[0]-p2[0]) - math.atan2(p1[1]-p2[1], p1[0]-p2[0]))

    return  360 - (deg + 360 if deg < 0 else deg)


def left_bot_point(left_half: List[Point], curr_y: float, curr_left: Point, curr_right: Point, xmid):
    """
    Given a left half list of points, a current y intercept, a maybe current left and right point hulls tangent and a the x of the division line 
    Find the left lowest point which is the one with the highest y intercept
    return that left point and the y intercept to the seperation line between both halfs
    """
    for lidx, lpt in enumerate(left_half):
        new_y = get_yslope(lpt, curr_right, xmid)
        if new_y > curr_y:
            curr_y = new_y
            curr_left = lpt
            curr_lidx = lidx
    
    return [curr_left, curr_y, curr_lidx]

def right_bot_point(right_half: List[Point], curr_y: float, curr_left: Point, curr_right: Point, xmid):
    """
    Given a right half list of points, a current y intercept, a maybe current left and right point hulls tangent and a the x of the division line 
    Find the right lowest point which is the one with the highest y intercept
    return that right point and the y intercept to the seperation line between both halfs
    """
    for ridx, rpt in enumerate(right_half):
        new_y = get_yslope(curr_left, rpt, xmid)
        if new_y > curr_y:
            curr_y = new_y
            curr_right = rpt
            curr_ridx = ridx
    
    return [curr_right, curr_y, curr_ridx]

def bot_tangent(left_half: List[Point], right_half: List[Point], curr_y: float, curr_left: Point, curr_right: Point, xmid):
    """
    Given a left and right half list of points, a current y intercept, a maybe current left and right point hulls tangent and a the x of the division line 
    Find the bot tangent
    return the 2 hull points from each half part of that tangent
    """
    base_y = curr_y
    
    lidx_yslope = left_bot_point(left_half, curr_y, curr_left, curr_right, xmid)
    curr_left = lidx_yslope[0]
    curr_yl = lidx_yslope[1]

    ridx_yslope = right_bot_point(right_half, curr_yl, curr_left, curr_right, xmid)
    curr_right = ridx_yslope[0]
    curr_yr = ridx_yslope[1]

    if curr_yr != base_y:
        return bot_tangent(left_half, right_half, curr_yr, curr_left, curr_right, xmid)
    else:
        return [curr_left, curr_right, lidx_yslope[2], ridx_yslope[2]]

    
def halfs_hull_bot_pts(left_half: List[Point], right_half: List[Point], hull_lptl, hull_rptl, xmid) -> List[Point]:
    """
    Given a list of points
    find the y slope at for these points and get the lower tangent with the highest y by calling bot_tangent
    return the points
    """
    
    curr_y = get_yslope(hull_lptl, hull_rptl, xmid)
    hull = bot_tangent(left_half, right_half, curr_y, hull_lptl, hull_rptl, xmid)
    return hull


def left_top_point(left_half: List[Point], curr_y: float, curr_left: Point, curr_right: Point, xmid):
    """
    Given a left half, a y intercept to the midline, a left hull point, its index, a right hull point and the x of the midline.
    Find the left highest point which is the one with the lowest y intercept
    return that left point and the y intercept on the seperation line between both halfs
    """
    for lidx, lpt in enumerate(left_half):
        new_y = get_yslope(lpt, curr_right, xmid)
        if new_y < curr_y:
            curr_y = new_y
            curr_left = lpt
            curr_lidx = lidx

    return [curr_left, curr_y, curr_lidx]

def right_top_point(right_half, curr_y: float, curr_left, curr_right, xmid):
    """
    Given a right half list of points, a current y intercept, a maybe current left and right point hulls tangent and a the x of the division line 
    Find the right highest point which is the one with the lowest y intercept
    return that right point and the y intercept to the seperation line between both halfs
    """
    for ridx, rpt in enumerate(right_half):
        new_y = get_yslope(curr_left, rpt, xmid)
        if new_y < curr_y:
            curr_y = new_y
            curr_right = rpt
            curr_ridx = ridx

    return [curr_right, curr_y, curr_ridx]

def top_tangent(left_half, right_half, curr_y, curr_left, curr_right, xmid) -> List[Point]:
    """
    Given a left and right half list of points, a current y intercept, a maybe current left and right point hulls tangent and a the x of the division line 
    Find the top tangent
    return the 2 hull points from each half part of that tangent
    """
    base_y = curr_y

    lidx_yslope = left_top_point(left_half, curr_y, curr_left, curr_right, xmid)
    curr_left = lidx_yslope[0]
    curr_yl = lidx_yslope[1]

    ridx_yslope = right_top_point(right_half, curr_yl, curr_left, curr_right, xmid)
    curr_right = ridx_yslope[0]
    curr_yr = ridx_yslope[1]
    
    if curr_yr != base_y:
        return top_tangent(left_half, right_half, curr_yr, curr_left, curr_right, xmid)
    else: 
        return [curr_left, curr_right, lidx_yslope[2], ridx_yslope[2]]


def get_yslope(lpt: Point, rpt: Point, x_line) -> float:
    """
    Given a left half point and a right half point.
    find the y intercept to that x mid line
    return their indexes
    """
    y_lr = y_intercept(lpt, rpt, x_line)
    return y_lr

def mid_lpt_rpt(left_half: List[Point], right_half: List[Point]) -> List[Point]:
    """
    Given a left half and a right half list of points.
    find the highest x point from the left half and the lowest x point from the right half.
    return them.
    """
    lpt = highes_x(left_half)
    rpt = lowes_x_index(right_half)

    return [lpt, rpt]

def get_xmid(lpt: Point, rpt: Point):
    """
    Given a left half and a right half point.
    return the x midline between both
    """
    return float((lpt[0] + rpt[0]) / 2)


def highes_y_index(points: List[Point]) -> int:
    """
    Given a list of points
    return the index of the point with the highest y value
    """
    y_max = points[0][1]
    result = 0
    for i, point in enumerate(points):
        if point[1] > y_max:
            y_max = point[1]
            result = i
    return result

def highes_x(points: List[Point]) -> int:
    """
    Given a list of points.
    return the point with the highest x value.
    """
    x_max = points[0][0]
    for point in points:
        if point[0] > x_max:
            x_max = point[0]
    return x_max

def lowes_x_index(points: List[Point]) -> int:
    """
    Given a list of points.
    return the index of the point with the highest x value.
    """
    x_min = points[0][0]
    for i, point in enumerate(points):
        if point[0] < x_min:
            x_min = point[0]

    return x_min

def hull_seg(hull_point1: Point, hull_point2: Point, points: List[Point], new_points = List[Point] ):
    """
    Given two hull points, a list of points, a list of points and hull points.
    find the next adjacent hull point in a clockwise manner, remove that point from list of points, add it to a hull point list and return it
    Note: this function modifies its argument.
    """
    hull_point3_idx = 0
    degR = angleR(hull_point1, hull_point2, points[0])
    for i, point in enumerate(points):
        new_degR = angleR(hull_point1, hull_point2, point)
        if new_degR > degR:
            hull_point3_idx = i
            degR = new_degR
    new_points.append(points[hull_point3_idx])
    points.pop(hull_point3_idx)
    
    return

def halfs_hull_pts(left_half, right_half):
    """
    Given a left half and a right half list of points.
    Find the rightest left points, the leftest right point and the x line seperating both halfs.
    find the y slope at for these points at that x mid and get the upper tangent with the lowest y
    return them
    """ 
    l_r_pts = mid_lpt_rpt(left_half, right_half)
    xmid = get_xmid(l_r_pts[0], l_r_pts[1])
    hull_lptl = l_r_pts[0]
    hull_rptl = l_r_pts[1]

    tophull = halfs_hull_top_pts(left_half, right_half, hull_lptl, hull_rptl, xmid)
    bothull = halfs_hull_bot_pts(left_half, right_half, hull_lptl, hull_rptl, xmid)
    return [tophull, bothull]

def halfs_hull_top_pts(left_half, right_half, hull_lptl, hull_rptl, xmid) -> List[int]:
    """
    Given a left half and a right half list of points.
    find the y slope at for these points at that x mid and get the upper tangent with the lowest y
    return them
    """
    curr_y = get_yslope(hull_lptl, hull_rptl, xmid)
    hull = top_tangent(left_half, right_half, curr_y, hull_lptl, hull_rptl, xmid)

    return hull