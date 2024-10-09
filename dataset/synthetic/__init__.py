from dataset.synthetic.markovian_diffeomorphic.chain_lin_3 import chain_lin_3_init
from dataset.synthetic.markovian_diffeomorphic.chain_nlin_3 import chain_nlin_3_init
from dataset.synthetic.markovian_diffeomorphic.chain_lin_4 import chain_lin_4_init
from dataset.synthetic.markovian_diffeomorphic.chain_lin_5 import chain_lin_5_init
from dataset.synthetic.markovian_diffeomorphic.collider_lin import collider_lin_init
from dataset.synthetic.markovian_diffeomorphic.fork_lin import fork_lin_init
from dataset.synthetic.markovian_diffeomorphic.fork_nlin import fork_nlin_init
from dataset.synthetic.markovian_diffeomorphic.largebd_nlin import largebd_nlin_init
from dataset.synthetic.markovian_diffeomorphic.simpson_nlin import simpson_nlin_init
from dataset.synthetic.markovian_diffeomorphic.simpson_symprod import simpson_symprod_init
from dataset.synthetic.markovian_diffeomorphic.triangle_lin import triangle_lin_init
from dataset.synthetic.markovian_diffeomorphic.triangle_nlin import triangle_nlin_init

from dataset.synthetic.recursive_continuous.back_door import back_door_init
from dataset.synthetic.recursive_continuous.front_door import front_door_init
from dataset.synthetic.recursive_continuous.m import m_init
from dataset.synthetic.recursive_continuous.napkin import napkin_init

from dataset.synthetic.regional_canonical.fairness import fairness_init
from dataset.synthetic.regional_canonical.fairness_xw import fairness_xw_init
from dataset.synthetic.regional_canonical.fairness_xy import fairness_xy_init
from dataset.synthetic.regional_canonical.fairness_yw import fairness_yw_init

from dataset.synthetic.dataset import ObservationalDataset

datasets = {
    # Markov Continuous
    'chain_lin_3': chain_lin_3_init,
    'chain_lin_4': chain_lin_4_init,
    'chain_lin_5': chain_lin_5_init,
    'chain_nlin_3': chain_nlin_3_init,
    'collider_lin': collider_lin_init,
    'fork_lin': fork_lin_init,
    'fork_nlin': fork_nlin_init,
    'largebd_nlin': largebd_nlin_init,
    'simpson_nlin': simpson_nlin_init,
    'simpson_symprod': simpson_symprod_init,
    'triangle_lin': triangle_lin_init,
    'triangle_nlin': triangle_nlin_init,
    # Recursive Continuous
    'back_door': back_door_init,
    'front_door': front_door_init,
    'm': m_init,
    'napkin': napkin_init,
    # Regional Canonical
    'fairness': fairness_init,
    'fairness_xw': fairness_xw_init,
    'fairness_xy': fairness_xy_init,
    'fairness_yw': fairness_yw_init,
}
