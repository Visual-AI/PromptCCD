from .promptccd_training_w_gmp import PromptCCD_Model as PromptCCD_Model_w_gmp_known_K
from .promptccd_training_w_gmp_unknown_K import PromptCCD_Model as PromptCCD_Model_w_gmp_unknown_K
from .promptccd_training_w_l2p import PromptCCD_Model as PromptCCD_Model_w_l2p_known_K
from .promptccd_training_w_dp import PromptCCD_Model as PromptCCD_Model_w_dp_known_K

model_dict = {
    'PromptCCD_w_GMP_known_K': PromptCCD_Model_w_gmp_known_K,
    'PromptCCD_w_GMP_unknown_K': PromptCCD_Model_w_gmp_unknown_K,
    'PromptCCD_w_L2P_known_K': PromptCCD_Model_w_l2p_known_K,
    'PromptCCD_w_DP_known_K': PromptCCD_Model_w_dp_known_K,
    'None': None,
}

def get_model(args):
    return {
        'ccd_model': model_dict[args.get('ccd_model', 'None')],
    }