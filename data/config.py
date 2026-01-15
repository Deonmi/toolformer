# config.py

MODEL_NAME = "your-model-name"

# Some hyperparameters in the sampling phase of Toolformer
TAU_S = 0.2      # Sampling threshold τ_s
TOP_K_POS = 5    # Each sample can only retain up to k positions
MAX_CALLS_PER_POS = 1  # Each location can have a maximum of m API calls

API_START_TOKEN = "["   # Or "[" "<API>", modify according to your own settings

# How many top candidates are returned by logprobs
TOP_LOGPROBS = 5
