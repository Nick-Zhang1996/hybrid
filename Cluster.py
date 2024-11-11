import numpy as np

class Cluster:
    def __init__(self, elements, distance_map, final_group = 3, final_distance = 0.2):
        self.clusters = [[i] for i in range(len(elements))]
        # initially each element has a cluster
        self.distance_map = distance_map
        self.final_group = final_group
        self.final_distance = final_distance

    def merge(self):
        min_dist = 0.0
        while len(self.clusters) >= self.final_group or min_dist < self.final_distance :
            # find smallest distance between clusters, merge
            min_dist = np.inf
            pair = None
            for i in range(len(self.clusters)):
                for j in range(i):
                    dist = self.clusterDistance(i,j)
                    if (dist < min_dist):
                        min_dist = dist
                        pair = (i,j)
            if (pair is None):
                break
            self.mergeClusters(*pair)

    def clusterDistance(self, i, j):
        # expected distance between clusters
        dist_vec = []
        for val1 in self.clusters[i]:
            for val2 in self.clusters[j]:
                dist_vec.append( self.distance_map[val1,val2] )
        return np.mean(dist_vec)

    def mergeClusters(self, i, j):
        self.clusters[i] += self.clusters[j]
        self.clusters.pop(j)
    def get(self):
        return self.clusters


if __name__=="__main__":
    elements_1 = np.random.multivariate_normal( [0,0] ,np.diag([0.3]*2), 10)
    elements_2 = np.random.multivariate_normal( [1,1] ,np.diag([0.3]*2), 10)
    elements = np.vstack([elements_1, elements_2])
    distance_map = np.inf * np.ones((elements.shape[0], elements.shape[0]))
    for i in range(elements.shape[0]):
        for j in range(i):
            distance_map[i,j] = distance_map[j,i] = np.linalg.norm( elements[i] - elements[j] )

    cluster = Cluster(elements, distance_map)
    cluster.merge()
    for group in cluster.clusters:
        val = [elements[i] for i in group]
        print(np.mean(val,axis=0))





