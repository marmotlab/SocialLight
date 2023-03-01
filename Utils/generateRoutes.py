import numpy as np
import sys
import time

top_entry = ['np15_nt21', 'np14_nt22', 'np13_nt23', 'np12_nt24', 'np11_nt25']
top_exit = ['nt21_np15', 'nt22_np14', 'nt23_np13', 'nt24_np12', 'nt25_np11']

left_entry = ['np16_nt21', 'np17_nt16', 'np18_nt11', 'np19_nt6', 'np20_nt1']
left_exit = ['nt21_np16', 'nt16_np17', 'nt11_np18', 'nt6_np19', 'nt1_np20']

right_entry = ['np10_nt25', 'np9_nt20', 'np8_nt15', 'np7_nt10', 'np6_nt5']
right_exit = ['nt25_np10', 'nt20_np9', 'nt15_np8', 'nt10_np7', 'nt5_np6']

bottom_entry = ['np1_nt1', 'np2_nt2', 'np3_nt3', 'np4_nt4', 'np5_nt5']
bottom_exit = ['nt1_np1', 'nt2_np2', 'nt3_np3', 'nt4_np4', 'nt5_np5']



# defined in km/h
MAX_SPEED = 10


def getClosestVector(scale, simplex):
    closestVector = np.zeros_like(simplex)
    ticker = 0
    currEntry = 0
    CDF = np.cumsum(simplex)
    for i in np.arange(scale):
        if ticker > CDF[currEntry]:
            currEntry += 1
        closestVector[currEntry] += 1
        ticker += (1 / (scale + 1))

    return closestVector


def getBadEntries():
    # 0/bad, 1/good
    M = [
        # Top entries
        list(np.zeros(10)) + list(np.ones(10)),
        list(np.zeros(5)) + list(np.ones(15)),
        list(np.zeros(5)) + list(np.ones(15)),
        list(np.zeros(5)) + list(np.ones(15)),
        list(np.zeros(5)) + list(np.ones(5)) + list(np.zeros(5)) + list(np.ones(5)),

        # Left entries
        list(np.zeros(10)) + list(np.ones(10)),
        list(np.ones(5)) + list(np.zeros(5)) + list(np.ones(10)),
        list(np.ones(5)) + list(np.zeros(5)) + list(np.ones(10)),
        list(np.ones(5)) + list(np.zeros(5)) + list(np.ones(10)),
        list(np.ones(5)) + list(np.zeros(5)) + list(np.ones(5)) + list(np.zeros(5)) ,

        # Right entries
        list(np.zeros(5)) + list(np.ones(5)) + list(np.zeros(5)) + list(np.ones(5)),
        list(np.ones(10)) + list(np.zeros(5)) + list(np.ones(5)),
        list(np.ones(10)) + list(np.zeros(5)) + list(np.ones(5)),
        list(np.ones(10)) + list(np.zeros(5)) + list(np.ones(5)),
        list(np.ones(10)) + list(np.zeros(10)),

        # Bottom entries
        list(np.ones(5)) + list(np.zeros(5)) + list(np.ones(5)) + list(np.zeros(5)),
        list(np.ones(15)) + list(np.zeros(5)),
        list(np.ones(15)) + list(np.zeros(5)),
        list(np.ones(15)) + list(np.zeros(5)),
        list(np.ones(10)) + list(np.zeros(10)),
    ]

    return M


def generateRoutes(outputFile, numCars, startTime, endTime, variedDistribution=False, episodeNumber=None):
    numEntryPoints = 20
    numExitPoints = numEntryPoints
    badEntries = getBadEntries()

    eps = 1e-10

    if variedDistribution:
        x = np.random.random()
        if x < 0:
            alpha = np.ones(20)
            beta = np.ones(20)
        else:
            print("Cars only coming from two roads")
            alpha = np.zeros(numEntryPoints) + eps
            beta = np.zeros(numExitPoints) + eps

            entryOne, entryTwo, exitOne, exitTwo = np.random.choice(4, 4, replace=False)

            for e in [entryOne, entryTwo]:
                scaledE = e * 3
                alpha[scaledE:scaledE + 3] = 100

            for e in [exitOne, exitTwo]:
                scaledE = e * 3
                beta[scaledE:scaledE + 3] = 100

    else:
        alpha = np.ones(20)
        beta = np.ones(20)
        # alpha = [100, 100, 100, 100, 100, 100] + ([eps] * 6)
        # beta = ([eps] * 6) + [100,100,100,100, 100, 100]

    # for cars coming from one side
    # alpha = [100, 100, 100] + ([eps] * 9)
    # beta = ([eps] * 9) + [100,100,100]

    # use these for cars coming from 2 roads
    # alpha = [100, 100, 100, 100, 100, 100] + ([eps] * 6)
    # beta = ([eps] * 6) + [100,100,100,100, 100, 100]

    # Use these for balanced case (cars coming from all 4 roads)

    # It's possible that the given scenario will not work.
    # In which case we want to generate entirely new parameters
    # and try again
    while True:
        numTries = 0
        maxTries = 100000
        start = np.random.dirichlet(alpha)
        end = np.random.dirichlet(beta)
        # start, end = np.random.dirichlet(alpha, 2)
        startVec = getClosestVector(numCars, start)
        endVec = getClosestVector(numCars, end)
        startCounter = np.zeros_like(startVec)
        endCounter = np.zeros_like(endVec)
        OD_matrix = np.zeros((numEntryPoints, numExitPoints))
        for i in range(numCars):
            while True:
                row = np.random.randint(0, numEntryPoints)
                col = np.random.randint(0, numExitPoints)
                if row == col:
                    continue

                if Ignore_bad_entry:
                    if startCounter[row] < startVec[row] and (endCounter[col] < endVec[col]):
                        OD_matrix[row, col] += 1
                        startCounter[row] += 1
                        endCounter[col] += 1
                        break
                    numTries += 1

                else:
                    if startCounter[row] < startVec[row] and (endCounter[col] < endVec[col]) \
                            and (badEntries[row][col] != 0):
                        OD_matrix[row, col] += 1
                        startCounter[row] += 1
                        endCounter[col] += 1
                        break
                    numTries += 1

                if numTries > maxTries:
                    break
        if numTries <= maxTries:
            break

    entryToSumo = [
        'np15_nt21', 'np14_nt22', 'np13_nt23', 'np12_nt24', 'np11_nt25',
        'np16_nt21', 'np17_nt16', 'np18_nt11', 'np19_nt6', 'np20_nt1',
        'np10_nt25', 'np9_nt20', 'np8_nt15', 'np7_nt10', 'np6_nt5',
        'np1_nt1', 'np2_nt2', 'np3_nt3', 'np4_nt4', 'np5_nt5',
    ]

    exitToSumo = [
        'nt21_np15', 'nt22_np14', 'nt23_np13', 'nt24_np12', 'nt25_np11',
        'nt21_np16', 'nt16_np17', 'nt11_np18', 'nt6_np19', 'nt1_np20',
        'nt25_np10', 'nt20_np9', 'nt15_np8', 'nt10_np7', 'nt5_np6',
        'nt1_np1', 'nt2_np2', 'nt3_np3', 'nt4_np4', 'nt5_np5'
    ]

    with open(outputFile, 'w') as f:
        f.write("<routes>")
        f.write("""
        <vType id="car" type="passenger" length="5" accel="2.6" decel="4.5" sigma="0.5" 
               maxSpeed="{maxSpeed}"
        />
        """.format(maxSpeed=MAX_SPEED))

        template = """
        <flow id="{ID}" type="car" begin="{startTime}" end="{endTime}" number="{numCars}" from="{origin}" to="{destination}" departPos="random_free" departLane="best" departSpeed="max" arrivalPos ="5" />
        """

        currID = 0
        for r in range(OD_matrix.shape[0]):
            for c in range(OD_matrix.shape[1]):
                if Ignore_bad_entry:
                    if r == c or (OD_matrix[r, c] == 0):
                        continue
                else:
                    if r == c or (OD_matrix[r,c] == 0) or badEntries[r][c] == 0:
                        continue


                flow = template.format(ID=currID, startTime=startTime, endTime=endTime,
                                       numCars=int(OD_matrix[r, c]), origin=entryToSumo[r],
                                       destination=exitToSumo[c])
                currID += 1

                f.write(flow)
        f.write("</routes>")



if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Using default arguments")
        outputFile = "./Manhattan_map/data_manhattan/dynamicRoute.rou.xml"
        numCars = 1500
        startTime = 0
        endTime = 10

        print("outputFile: {} \nnumCars: {} \nstartTime: {} \nendTime: {}".format(outputFile, numCars, startTime,
                                                                                  endTime))
    else:
        outputFile = sys.argv[1]
        numCars = int(sys.argv[2])
        startTime = int(sys.argv[3])
        endTime = int(sys.argv[4])

    # dynamicRoute.rou.xml
    start = time.time()
    generateRoutes(outputFile, numCars, startTime, endTime)
    end = time.time()
    print("time:{}".format(end-start))
