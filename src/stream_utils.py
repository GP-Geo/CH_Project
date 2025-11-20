import numpy as np

def outlet_node_ids_from_streampoi(s):
    """Return outlet *node IDs* from a StreamObject using streampoi('outlets')."""
    outlet_mask = s.streampoi('outlets')
    return np.flatnonzero(outlet_mask)