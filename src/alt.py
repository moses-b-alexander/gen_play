
# from * import *


# TODO develop streamlit frontend (xxx.py)

# TODO agentic hyperparameter tuning (hp_search.py)

# TODO incorporate T into sde (ai/dynamics.py)
# Dynamics.f/g -- concat sin(n*t) and cos(n*t)
#   to h before f0/g0, not addition; num_freqs = max(1, dim_embedding // 64),
#   time_enc_dim = 2 * num_freqs; no new HP, internal constant only;
#   deferred until after some real training)
