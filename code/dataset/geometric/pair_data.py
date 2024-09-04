from torch_geometric.data import Data

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_q':
            return self.x_q.size(0)
        if key == 'edge_index_r':
            return self.x_r.size(0)
        return super().__inc__(key, value, *args, **kwargs)