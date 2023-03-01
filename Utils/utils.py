import sys
import optparse
import traci
import os
from sumolib import checkBinary


def describe(text_description, infos):
    textual = text_description
    total = textual + '\n'
    for info in infos:
        total += info + '\n'
    return total


def writeConfigFile(configFilePath, routeFileName):
    configFileTemplate = """
    <configuration>
  <input>
    <net-file value="exp.net.xml"/>
    <route-files value="{routeFileName}"/>
    <additional-files value="exp.add1.xml"/>
  </input>

</configuration>
    """
    with open(configFilePath, 'w') as f:
        f.write(configFileTemplate.format(routeFileName=routeFileName))


def check_SUMO_HOME():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")


def Manhattan_traffic_light_phase_code():
    # phase codes based on cross.net.xml
    # sl stands for straight and left-turn
    PHASE_NS_GREEN = 0  # action 0
    PHASE_NS_YELLOW = 1
    PHASE_NSL_GREEN = 2  # action 1
    PHASE_NSL_YELLOW = 3
    PHASE_EW_GREEN = 4  # action 2
    PHASE_EW_YELLOW = 5
    PHASE_EWL_GREEN = 6  # action 3
    PHASE_EWL_YELLOW = 7
    PHASE_N_SL_GREEN = 8  # action 4
    PHASE_N_SL_YELLOW = 9
    PHASE_E_SL_GREEN = 10  # action 5
    PHASE_E_SL_YELLOW = 11
    PHASE_S_SL_GREEN = 12  # action 6
    PHASE_S_SL_YELLOW = 13
    PHASE_W_SL_GREEN = 14  # action 7
    PHASE_W_SL_YELLOW = 15
    PHASE_ALL_RED = 16

    return 0


def Manhattan_neighbor_map():
    neighbor_map = {}
    # corner nodes
    neighbor_map['nt1'] = ['nt6', 'nt2']
    neighbor_map['nt5'] = ['nt10', 'nt4']
    neighbor_map['nt21'] = ['nt22', 'nt16']
    neighbor_map['nt25'] = ['nt20', 'nt24']
    # edge nodes
    neighbor_map['nt2'] = ['nt7', 'nt3', 'nt1']
    neighbor_map['nt3'] = ['nt8', 'nt4', 'nt2']
    neighbor_map['nt4'] = ['nt9', 'nt5', 'nt3']
    neighbor_map['nt22'] = ['nt23', 'nt17', 'nt21']
    neighbor_map['nt23'] = ['nt24', 'nt18', 'nt22']
    neighbor_map['nt24'] = ['nt25', 'nt19', 'nt23']
    neighbor_map['nt10'] = ['nt15', 'nt5', 'nt9']
    neighbor_map['nt15'] = ['nt20', 'nt10', 'nt14']
    neighbor_map['nt20'] = ['nt25', 'nt15', 'nt19']
    neighbor_map['nt6'] = ['nt11', 'nt7', 'nt1']
    neighbor_map['nt11'] = ['nt16', 'nt12', 'nt6']
    neighbor_map['nt16'] = ['nt21', 'nt17', 'nt11']
    # internal nodes
    for i in [7, 8, 9, 12, 13, 14, 17, 18, 19]:
        n_node = 'nt' + str(i + 5)
        s_node = 'nt' + str(i - 5)
        w_node = 'nt' + str(i - 1)
        e_node = 'nt' + str(i + 1)
        cur_node = 'nt' + str(i)
        neighbor_map[cur_node] = [n_node, e_node, s_node, w_node]
    return neighbor_map


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def make_gif(gif_folder, fname):
    command = 'ffmpeg -framerate 5 -i "{tempGifFolder}/step%03d.png" {outputFile}'.format(tempGifFolder=gif_folder,
                                                                                          outputFile=fname)

    os.system(command)

    deleteTempImages = "rm {tempGifFolder}/*".format(tempGifFolder=gif_folder)
    os.system(deleteTempImages)
    print("wrote gif")


if __name__ == '__main__':
    pass
