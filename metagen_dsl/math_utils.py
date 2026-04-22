import numpy as np
from dataclasses import dataclass
import math 

def fp_equals(a:float, b:float, eps:float=1e-4) -> bool:
    return abs(a-b) < eps

def array_fp_equals(a:np.array, b:np.array, eps:float=1e-4) -> bool:
    aflat = a.flatten()
    bflat = b.flatten()
    if aflat.size != bflat.size:
        return False
    for eid in range(aflat.size):    
        if abs(aflat[eid]-bflat[eid]) > eps:
            return False
    return True

def is_power_of_2(x:float) -> bool: 
    logRepeats = math.log(x, 2)
    return math.floor(logRepeats) == logRepeats

def clamp(val:float, minval:float, maxval:float) -> float:
    if val > maxval:
        return maxval
    elif val < minval:
        return minval
    return val

def projAOntoB(a:np.array, b:np.array):
    assert a.shape == b.shape, "a and b must live in the same vector space"
    return np.dot(a, b) / np.dot(b, b) * b

# inspired by response https://math.stackexchange.com/a/61073 to question https://math.stackexchange.com/questions/61061/line-plane-intersection-in-high-dimension
def nd_line_segment_intersection(a0:np.array, a1:np.array, b0:np.array, b1:np.array) -> bool:
    assert a0.size == a1.size == b0.size == b1.size, "All points must live in the same space"
    starting_dim:int = a0.size
    if starting_dim == 3: 
        info:segmentPairIntersectionInfo = minDistanceBetweenLineSegmentsInR3(a0, a1, b0, b1)
        return info.ABintersect
    
    # get vectors by selecting one of the endpoints and subtracting it from all other points
    p_ref = a0
    v0 = a1 - p_ref
    v1 = b0 - p_ref
    v2 = b1 - p_ref

    R3 = np.mat([v0, v1, v2])
    assert R3.shape[0] == 3 and R3.shape[1] == starting_dim, "R3 matrix not incorrect"
    R3_T = np.transpose(R3)
    
    detR3 = np.linalg.det(R3 * R3_T)
    if not fp_equals(detR3, 0):
        return False            # lines don't live in the same 3d subspace, can't intersect
    
    def to_R3(x:np.array):
        xr3_col = R3* np.transpose(np.atleast_2d(x-p_ref))
        return np.array([xr3_col[0, 0], xr3_col[1, 0], xr3_col[2,0]]) # return a 1x3, must be array for minDistanceBetweenLineSegmentsInR3 input

    def to_Rd(x:np.array): # NOTE: won't work for a singular matrix (e.g., lines really live in the same 2d space). If you want to use this, it'll have to be generalized.
        xrd_col = R3_T * np.linalg.inv(R3 * R3_T) * np.transpose(np.atleast_2d(x)) + np.transpose(np.atleast_2d(p_ref))    
        return np.transpose(xrd_col) # return a 1xd

    info:segmentPairIntersectionInfo = minDistanceBetweenLineSegmentsInR3(to_R3(a0), to_R3(a1), to_R3(b0), to_R3(b1))
    #     closestPtOnA = to_Rd(info.closestPtOnA)
    #     closestPtOnB = to_Rd(info.closestPtOnB)
    return info.ABintersect
@dataclass
class segmentPairIntersectionInfo:
    ABintersect:bool
    ABparallel:bool
    closestPtOnA:np.array
    closestPtOnB:np.array
    multipleClosestPts:bool
    minDistance:float

# inspired by https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
def minDistanceBetweenLineSegmentsInR3(a0:np.array, a1:np.array, b0:np.array, b1:np.array) -> segmentPairIntersectionInfo:
    assert a0.size == 3 and a1.size == 3 and b0.size == 3 and b1.size == 3, "All input positions must be in 3D space"

    intersectionFound = False
    segmentsParallel = False
    multipleClosestPts = False
    closestPtOnA = None
    closestPtOnB = None
    minDistance = -1

    A:np.array = a1 - a0
    magA:float = np.linalg.norm(A)
    unitA:np.array = A / magA

    B:np.array = b1 - b0
    magB:float = np.linalg.norm(B)
    unitB:np.array = B / magB

    cross:np.array = np.cross(unitA, unitB)
    sqrMagCross:float = np.linalg.norm(cross) ** 2

    # special case: parallel lines if sqrMagCross = 0
    if fp_equals(sqrMagCross, 0):
        segmentsParallel = True
        # make sure that a and b go in the same direction along unitA
        _b0:np.array = b0 
        _b1:np.array = b1
        if np.dot(unitA, unitB) < 0: # opposite directions, flip b0, b1
            _b0 = b1
            _b1 = b0
        
        d0:float = np.dot(unitA, _b0-a0)
        d1:float = np.dot(unitA, _b1-a0)

        # segment B is either fully before or after segment A along direction unitA
        if (d0 <= 0 and d1 <= 0) or (d0 >= magA and d1 >= magA):
            closestPtOnA = a0 if d0 <= 0 else a1 # if B before A, then a0 closer, else a1 closer
            closestPtOnB = _b1 if d0 <= 0  else _b0 # if B before A, b1 closer, else b0
        else:  # segments are overlapping along direction unitA -- 4 cases (all have >1 closestpts, we pick one valid one)
            multipleClosestPts = True
            if (d0 <= 0 and d1 > 0 and d1 <= magA): # starts before, ends inside
                closestPtOnA = a0 + d1 * unitA
                closestPtOnB = _b1
            elif (d0 <= 0 and d1 >= magA): # starts before, ends after
                closestPtOnA = a0
                _unitB:np.array = _b1 - _b0
                _unitB = _unitB / np.linalg.norm(_unitB)
                t:float = np.dot(_unitB, a0 - _b0)
                closestPtOnB = _b0 + t * _unitB
            else: # starts inside, ends after OR starts inside, ends inside
                closestPtOnA = a0 + d0 * unitA
                closestPtOnB = _b0
        minDistance = np.linalg.norm(closestPtOnA - closestPtOnB)
        intersectionFound = fp_equals(minDistance, 0)
        return segmentPairIntersectionInfo(intersectionFound, segmentsParallel, closestPtOnA, closestPtOnB, multipleClosestPts, minDistance)

    # else we know their closest distance occurs between two points located along the lines' cross product
    interLineVec:np.array = b0 - a0
    matA = np.mat([interLineVec, unitB, cross])
    detA:float = np.linalg.det(matA)
    matB = np.mat([interLineVec, unitA, cross])
    detB:float = np.linalg.det(matB)

    t0:float = detA / sqrMagCross
    t1:float = detB / sqrMagCross
 
    if (t0 < 0):   # clamp to A if necessary
        closestPtOnA = a0
    elif (t0 > magA):
        closestPtOnA = a1
    else:
        closestPtOnA = a0 + (unitA * t0) # projected closest pt on A

    if (t1 < 0):  # clamp to B if necessary
         closestPtOnB = b0
    elif (t1 > magB):
        closestPtOnB = b1
    else:
        closestPtOnB = b0 + (unitB * t1) # projected closest pt on B

    if (t0 < 0 or t0 > magA): # update pB if necessary
        proj:float = np.dot(unitB, closestPtOnA-b0)
        proj = clamp(proj, 0, magB)
        closestPtOnB = b0 + proj * unitB

    if (t1 < 0 or t1 > magB):
        proj = np.dot(unitA, closestPtOnB - a0)
        proj = clamp(proj, 0, magA)
        closestPtOnA = a0 + proj * unitA
    minDistance = np.linalg.norm(closestPtOnA - closestPtOnB)
    intersectionFound = fp_equals(minDistance, 0)
    return segmentPairIntersectionInfo(intersectionFound, segmentsParallel, closestPtOnA, closestPtOnB, multipleClosestPts, minDistance)


if __name__ == "__main__":
    a0 = np.array([0.0, 0.0, 0.0])
    a1 = np.array([0.0, 0.0, 1.0])
    b0 = np.array([0.0, 1.0, 1.0])
    b1 = np.array([0.0, 1.0, 0.0])

    print("intersection: " + str(minDistanceBetweenLineSegmentsInR3(a0, a1, b0, b1).ABintersect))

    a0 = np.array([1.0, 0.0, 0.0, 0.0])
    a1 = np.array([1.0, 0.0, 0.0, 1.0])
    b0 = np.array([1.0, 0.0, 1.0, 1.0])
    b1 = np.array([1.0, 0.0, 1.0, 0.0])

    print("intersection: " + str(nd_line_segment_intersection(a0, a1, b0, b1)))