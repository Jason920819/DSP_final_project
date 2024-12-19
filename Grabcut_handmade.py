import numpy as np
import cv2
import networkx as nx
from scipy.linalg import inv, det
from sklearn.cluster import KMeans

class GMM:
    components_count = 5

    def __init__(self, model=None, image=None):
        model_size = 3 + 9 + 1  # Mean (3), Covariance (9), Weight (1)
        if model is None:
            self.model = np.zeros((self.components_count, model_size))
            if image is not None:
                self._initialize_with_kmeans(image)
            else:
                raise ValueError("Image data is required for KMeans initialization.")
        else:
            self.model = model

        self.coefs = self.model[:, -1]
        self.means = self.model[:, :3]
        self.covariances = self.model[:, 3:12].reshape((-1, 3, 3))
        self.inverted_covs = np.zeros_like(self.covariances)
        self.determinants = np.zeros(self.components_count)
        self.sums = np.zeros((self.components_count, 3))
        self.prods = np.zeros((self.components_count, 3, 3))
        self.sample_counts = np.zeros(self.components_count, dtype=int)

        for i in range(self.components_count):
            if self.coefs[i] > 0:
                self._calc_inverse_and_determinant(i)

    def _initialize_with_kmeans(self, image):
        # Flatten the image to a 2D array of pixels
        h, w, c = image.shape
        pixels = image.reshape(-1, c).astype(np.float64)

        # Apply KMeans to cluster pixels into components_count clusters
        kmeans = KMeans(n_clusters=self.components_count, random_state=42).fit(pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Initialize GMM means with KMeans centers
        self.model[:, :3] = centers

        # Calculate initial covariances and weights
        for i in range(self.components_count):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) > 1:
                self.model[i, 3:12] = np.cov(cluster_pixels, rowvar=False).flatten()
            else:
                self.model[i, 3:12] = np.eye(3).flatten()  # Default to identity matrix for small clusters
            self.model[i, -1] = len(cluster_pixels) / len(pixels)  # Weight proportional to cluster size

    def __call__(self, color, component=None):
        if component is None:
            return sum(self.coefs[i] * self._gaussian_prob(i, color) for i in range(self.components_count))
        else:
            return self._gaussian_prob(component, color)

    def which_component(self, color):
        return np.argmax([self._gaussian_prob(i, color) for i in range(self.components_count)])

    def init_learning(self):
        self.sums.fill(0)
        self.prods.fill(0)
        self.sample_counts.fill(0)

    def add_sample(self, component, color):
        self.sums[component] += color
        self.prods[component] += np.outer(color, color)
        self.sample_counts[component] += 1

    def end_learning(self):
        total_count = np.sum(self.sample_counts)
        for i in range(self.components_count):
            if self.sample_counts[i] > 0:
                inv_count = 1.0 / self.sample_counts[i]
                self.means[i] = self.sums[i] * inv_count
                self.covariances[i] = self.prods[i] * inv_count - np.outer(self.means[i], self.means[i])
                self.coefs[i] = self.sample_counts[i] / total_count
                self._calc_inverse_and_determinant(i)

    def _calc_inverse_and_determinant(self, i):
        try:
            self.inverted_covs[i] = inv(self.covariances[i])
            self.determinants[i] = max(det(self.covariances[i]), 1e-10)  # Add lower bound
        except np.linalg.LinAlgError:
            self.inverted_covs[i] = np.eye(3)
            self.determinants[i] = 1e-10

    def _gaussian_prob(self, i, color):
        diff = color - self.means[i]
        exponent = -0.5 * diff.dot(self.inverted_covs[i]).dot(diff)
        normalization = (2 * np.pi) ** -1.5 * (self.determinants[i] + 1e-10) ** -0.5
        return max(normalization * np.exp(exponent), 1e-10)  # Add lower bound

# The rest of the code (functions like calc_beta, calc_n_weights, grabcut, etc.) remains unchanged.

def calc_beta(image):
    beta = 0
    h, w = image.shape[:2]
    for y in range(h):
        for x in range(w):
            color = image[y, x].astype(np.float64)
            if x > 0:
                diff = color - image[y, x - 1].astype(np.float64)
                beta += np.dot(diff, diff)
            if y > 0:
                diff = color - image[y - 1, x].astype(np.float64)
                beta += np.dot(diff, diff)
    beta = 1.0 / (2 * beta / (4 * h * w - 3 * h - 3 * w + 2))
    return beta

def calc_n_weights(image, beta, gamma):
    h, w = image.shape[:2]
    leftW = np.zeros((h, w))
    upW = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            color = image[y, x].astype(np.float64)
            if x > 0:
                diff = color - image[y, x - 1].astype(np.float64)
                leftW[y, x] = gamma * np.exp(-beta * np.dot(diff, diff))
            if y > 0:
                diff = color - image[y - 1, x].astype(np.float64)
                upW[y, x] = gamma * np.exp(-beta * np.dot(diff, diff))
    return leftW, upW

def init_mask_with_rect(mask, img_size, rect):
    mask.fill(0)  # GC_BGD
    x, y, w, h = rect
    mask[y:y + h, x:x + w] = 3  # GC_PR_FGD

def assign_gmm_components(image, mask, bgd_gmm, fgd_gmm, comp_idxs):
    h, w = image.shape[:2]
    for y in range(h):
        for x in range(w):
            color = image[y, x].astype(np.float64)
            if mask[y, x] in (0, 2):  # GC_BGD, GC_PR_BGD
                comp_idxs[y, x] = bgd_gmm.which_component(color)
            else:
                comp_idxs[y, x] = fgd_gmm.which_component(color)

def learn_gmm(image, mask, comp_idxs, bgd_gmm, fgd_gmm):
    bgd_gmm.init_learning()
    fgd_gmm.init_learning()
    h, w = image.shape[:2]
    for y in range(h):
        for x in range(w):
            color = image[y, x].astype(np.float64)
            component = comp_idxs[y, x]
            if mask[y, x] in (0, 2):  # GC_BGD, GC_PR_BGD
                bgd_gmm.add_sample(component, color)
            else:
                fgd_gmm.add_sample(component, color)
    bgd_gmm.end_learning()
    fgd_gmm.end_learning()

def construct_gc_graph(image, mask, bgd_gmm, fgd_gmm, lambda_, leftW, upW):
    h, w = image.shape[:2]
    graph = nx.DiGraph()
    source = 'source'
    sink = 'sink'
    graph.add_node(source)
    graph.add_node(sink)

    # 修改权重计算，避免无界流
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            color = image[y, x].astype(np.float64)
            
            # 计算源点和汇点的权重
            if mask[y, x] in (2, 3):  # GC_PR_BGD, GC_PR_FGD
                from_source = max(0.01, -np.log(bgd_gmm(color) + 1e-10))
                to_sink = max(0.01, -np.log(fgd_gmm(color) + 1e-10))
            elif mask[y, x] == 0:  # GC_BGD
                from_source = 0.01
                to_sink = lambda_
            else:  # GC_FGD
                from_source = lambda_
                to_sink = 0.01
            
            graph.add_edge(source, idx, capacity=from_source)
            graph.add_edge(idx, sink, capacity=to_sink)

    # 添加像素间的边
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if x > 0:
                neighbor_idx = y * w + (x - 1)
                weight = leftW[y, x]
                graph.add_edge(idx, neighbor_idx, capacity=weight)
                graph.add_edge(neighbor_idx, idx, capacity=weight)
            if y > 0:
                neighbor_idx = (y - 1) * w + x
                weight = upW[y, x]
                graph.add_edge(idx, neighbor_idx, capacity=weight)
                graph.add_edge(neighbor_idx, idx, capacity=weight)

    return graph

def estimate_segmentation(graph, mask):
    h, w = mask.shape
    source = 'source'
    sink = 'sink'
    
    # 使用networkx的最大流算法
    cut_value, partition = nx.minimum_cut(graph, source, sink)
    
    # 分区包含两个节点集合
    reachable, non_reachable = partition
    
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if mask[y, x] in (2, 3):  # GC_PR_BGD, GC_PR_FGD
                mask[y, x] = 3 if idx in reachable else 2

#原始版本:只能支援框長方形
""" def grabcut(image, mask, rect, bgd_model, fgd_model, iter_count):
    beta = calc_beta(image)
    gamma = 50
    lambda_ = 9 * gamma

    # Initialize GMM models
    bgdGMM = GMM(bgd_model)
    fgdGMM = GMM(fgd_model)

    # Calculate neighbor weights
    leftW, upW = calc_n_weights(image, beta, gamma)

    # Initialize mask with rectangle
    init_mask_with_rect(mask, image.shape[:2], rect)

    # Allocate component index array
    comp_idxs = np.zeros(image.shape[:2], dtype=np.int32)

    # Iterative GrabCut process
    for i in range(iter_count):
        print(f"Iteration {i+1}:")

        # Assign pixels to GMM components
        assign_gmm_components(image, mask, bgdGMM, fgdGMM, comp_idxs)
        print("Component indices assigned.")

        # Learn GMM models
        learn_gmm(image, mask, comp_idxs, bgdGMM, fgdGMM)
        print("GMM models updated.")
        print(f"Background GMM means:\n{bgdGMM.means}")
        print(f"Foreground GMM means:\n{fgdGMM.means}")

        # Construct graph
        graph = construct_gc_graph(image, mask, bgdGMM, fgdGMM, lambda_, leftW, upW)
        print("Graph constructed.")

        # Estimate segmentation
        estimate_segmentation(graph, mask)
        print("Mask updated.")
        print(" ")

    print("GrabCut completed.")
    print(f"Background GMM means:\n{bgdGMM.means}")
    print(f"Foreground GMM means:\n{fgdGMM.means}") """
    
##修改過後的版本，可以支援框長方形跟多邊形
def grabcut(image, mask, rect, bgd_model, fgd_model, iter_count):
    beta = calc_beta(image)
    gamma = 50
    lambda_ = 9 * gamma

    # Initialize GMM models
    bgdGMM = GMM(bgd_model)
    fgdGMM = GMM(fgd_model)

    # Calculate neighbor weights
    leftW, upW = calc_n_weights(image, beta, gamma)

    # Use mask directly if rect is None
    if rect is not None:
        init_mask_with_rect(mask, image.shape[:2], rect)
    else:
        if np.all(mask == 0):  # 確保遮罩不是全背景
            raise ValueError("Mask is not properly initialized. Ensure foreground/background areas are labeled.")

    # Allocate component index array
    comp_idxs = np.zeros(image.shape[:2], dtype=np.int32)

    # Iterative GrabCut process
    for i in range(iter_count):
        print(f"Iteration {i+1}:")

        # Assign pixels to GMM components
        assign_gmm_components(image, mask, bgdGMM, fgdGMM, comp_idxs)
        #print("Component indices assigned.")

        # Learn GMM models
        learn_gmm(image, mask, comp_idxs, bgdGMM, fgdGMM)
        #print("GMM models updated.")
        #print(f"Background GMM means:\n{bgdGMM.means}")
        #print(f"Foreground GMM means:\n{fgdGMM.means}")

        # Construct graph
        graph = construct_gc_graph(image, mask, bgdGMM, fgdGMM, lambda_, leftW, upW)
        #print("Graph constructed.")

        # Estimate segmentation
        estimate_segmentation(graph, mask)
        #print("Mask updated.")
        print(" ")

    print("GrabCut completed.")
    #print(f"Background GMM means:\n{bgdGMM.means}")
    #print(f"Foreground GMM means:\n{fgdGMM.means}")
