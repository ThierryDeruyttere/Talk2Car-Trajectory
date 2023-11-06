import torch
from utils.sample_mix import sample_mix
from utils.kmeans import kmeans


def sample_goals_TTST(waypoint_mix, num_goals, H, W):
    """Take num_goals - 1 samples from the cluster"""
    goal_samples_candidates, goal_samples_candidates_prob = sample_mix(
        waypoint_mix,
        num_samples=10000,
        H_scale=H,
        W_scale=W,
        output_in_wh=True,
        return_probs=True,
    )

    goal_samples_candidates = goal_samples_candidates[
        :, -1:, :, :
    ]  # B * num_paths, 1, 10000, 2
    goal_samples_candidates_prob = goal_samples_candidates_prob[
        :, -1:, :
    ]  # B * num_paths, 1, 10000

    goal_samples_candidates = goal_samples_candidates.permute(
        2, 0, 1, 3
    )  # 10000, B * num_paths, 1, 2 - reshaped for clustering
    num_clusters = num_goals - 1

    # Iterate through all person/batch_num, as this k-Means implementation doesn't support batched clustering
    goal_samples_list = []
    for sample_id in range(goal_samples_candidates.shape[1]):
        goal_sample = goal_samples_candidates[:, sample_id, 0]

        # Actual k-means clustering, Outputs:
        # cluster_ids_x -  Information to which cluster_idx each point belongs to
        # cluster_centers - list of centroids, which are our new goal samples
        cluster_ids_x, cluster_centers = kmeans(
            X=goal_sample,
            num_clusters=num_clusters,
            distance="euclidean",
            device=goal_sample.device,
            tqdm_flag=False,
            tol=0.001,
            iter_limit=1000,
        )  # num_clusters, 2
        goal_samples_list.append(cluster_centers)

    goal_samples_clustering = torch.stack(
        goal_samples_list
    )  # B * num_paths, num_clusters, 2
    goal_samples_clustering = goal_samples_clustering.permute(1, 0, 2).unsqueeze(
        2
    )  # num_clusters, B * num_paths, 1, 2

    """Now take one peak sample"""
    """ALTERNATIVE 1 - just sample it"""
    # goal_samples_peak = sample_mix(
    #     waypoint_mix, num_samples=1, H_scale=H, W_scale=W, output_in_wh=True
    # )[:, -1:, :, :]  # B * num_paths, 1, 1, 2
    # goal_samples_peak = goal_samples_peak.permute(2, 0, 1, 3)  # 1, B * num_paths, 1, 2

    # TAKE 10000 CANDIDATES, EVALUATE THEIR LOGPROBS AND TAKE THE ONE WITH HIGHEST LOG PROB"""
    """ALTERNATIVE 2 - take the most probable point"""
    goal_samples_candidates = goal_samples_candidates.permute(
        1, 2, 0, 3
    )  # B * num_paths, 1, 10000, 2
    # goal_samples_candidates_prob [B * num_paths, 1, 10000]
    best_candidate_ind = goal_samples_candidates_prob.max(-1)[1]  # [B * num_paths, 1]
    goal_samples_peak = torch.gather(
        goal_samples_candidates,
        2,
        best_candidate_ind.unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, 2),
    )  # [B * num_paths, 1, 1, 2]
    goal_samples_peak = goal_samples_peak.permute(2, 0, 1, 3)  # 1, B * num_paths, 1, 2

    goal_samples = torch.cat(
        [goal_samples_peak, goal_samples_clustering], dim=0
    )  # num_goals, B * num_paths, 1, 2
    return goal_samples
