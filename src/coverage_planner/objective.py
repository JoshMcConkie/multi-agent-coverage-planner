import numpy as np

class Objective:
    def build_util_matrix():
        print("Please define objective")
        raise NotImplementedError

class Coverage(Objective):
    @staticmethod
    def build_util_matrix(paths,coordinates) -> np.ndarray:
        util_mat = np.zeros((len(paths),(len(coordinates))))
        for i,path in enumerate(paths):
            for j,c in enumerate(coordinates):
                # 1 represents coverage
                util_mat[i][j] = 1 if c in path else 0 
        return util_mat
    
    
