from params.Final.important_params import Environment

if Environment == 'Manhattan':
    from params.Final.Environment_params.Manhattan_params import *

elif Environment == 'CityFlowV2':
    from params.Final.Environment_params.CityFlowV2_params import *