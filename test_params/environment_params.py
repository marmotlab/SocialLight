params_path = 'Final'
from test_params.Final.important_params import Environment


if Environment == 'Manhattan':
    from test_params.Environment_params.Manhattan_params import *

elif Environment == 'Monaco':
    from test_params.Environment_params.Monaco_params import *

elif Environment == 'CityFlow':
    from test_params.Environment_params.CityFlow_params import *

elif Environment == 'CityFlowV2':
    from test_params.Environment_params.CityFlowV2_params import *