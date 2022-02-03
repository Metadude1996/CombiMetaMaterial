"Aleksi Bossart, 26.2.2019"
"code to generate periodic htiles X vtiles config for a given k X k config"

import numpy as np


def gen_config(sub_config, vtiles, htiles):
    sub_vdim, sub_hdim = np.shape(sub_config)
    aperio = np.zeros((vtiles*sub_vdim, htiles*sub_hdim), dtype=int)
    for vtilecount in range(vtiles):
        for htilecount in range(htiles):
            for vcellcount in range(sub_vdim):
                for hcellcount in range(sub_hdim):
                    aperio[(vtilecount*sub_vdim)+vcellcount, (htilecount*sub_hdim)+hcellcount] = sub_config[vcellcount,hcellcount]
    return aperio            