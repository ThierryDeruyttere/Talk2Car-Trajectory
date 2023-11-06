import torch
from utils.sample_mix import sample_mix
from utils.create_mix import multivariate_gaussian_params

import torch.distributions as D


def sample_waypoints_CWS(
    waypoint_mix,
    goal_samples,
    start_pos,
    num_trajs,
    num_goals,
    H,
    W,
    sigma_factor=6.0,
    ratio=2.0,
    rot=True,
    waypoints_ix=None
):

    goal_samples = goal_samples.repeat(
        num_trajs, 1, 1, 1
    )  # num_goals * num_traj, B * num_paths, 1, 2
    last_observed = start_pos[:, 0] * torch.tensor([W, H]).to(start_pos)  # B * num_paths, 2

    (
        all_waypoint_samples_candidates,
        all_waypoint_samples_candidates_probs,
    ) = sample_mix(
        waypoint_mix,
        num_samples=10000,
        H_scale=H,
        W_scale=W,
        output_in_wh=True,
        return_probs=True,
    )
    all_waypoint_samples_candidates = all_waypoint_samples_candidates[
        :, :-1, :, :
    ]  # [B * num_paths, num_waypoints - 1, 10000, 2]
    all_waypoint_samples_candidates_probs = all_waypoint_samples_candidates_probs[
        :, :-1, :
    ]  # [B * num_paths, num_waypoints - 1, 10000]

    waypoint_samples_list = (
        []
    )  # in the end this should be a list of [num_goals * num_traj, B * num_paths, num_waypoints, 2] waypoint coordinates
    for g_num, waypoint_samples in enumerate(
        goal_samples.squeeze(2)
    ):  # for each goal candidate
        # g_num in [0, num_traj-1] the goal_samples are the same - they were repeated
        # waypoint_samples [B * num_paths, 2] - We start with the goal, and move forwards towards begining
        waypoint_list = []  # for each K sample have a separate list
        waypoint_list.append(waypoint_samples)

        for waypoint_num in reversed(
            range(len(waypoints_ix) - 1)
        ):  # for each waypoint - e.g. 2nd, than 1st than 0th if waypoints_ix = [4, 9, 14, 19]
            distance = last_observed - waypoint_samples  # B * num_paths, 2
            waypoint_samples_candidates = all_waypoint_samples_candidates[
                :, waypoint_num
            ]  # [B * num_paths, 10000, 2]
            waypoint_samples_candidates_probs = all_waypoint_samples_candidates_probs[
                :, waypoint_num
            ]  # [B * num_paths, 10000]

            intermediate_distribution_locs = []
            intermediate_distribution_covs = []
            intermediate_distribution_chols = []
            traj_idx = (
                g_num // num_goals
            )  # idx of trajectory for the same goal - for 1st traj we use wide gaussian, for 2nd a bit tighter, for 3rd very tight gaussian
            for dist, coordinate in zip(
                distance, waypoint_samples
            ):  # for each dataset sample
                # dist [2], coordinate [2]
                length_ratio = 1 / (waypoint_num + 2)
                gauss_mean = coordinate + (
                    dist * length_ratio
                )  # 2, Get the intermediate point's location using CV model
                sigma_factor_ = sigma_factor - traj_idx
                (
                    intermediate_distribution_loc,
                    intermediate_distribution_cov,
                    intermediate_distribution_chol,
                ) = multivariate_gaussian_params(
                    gauss_mean, dist, sigma_factor_, ratio, rot=rot
                )
                intermediate_distribution_locs.append(intermediate_distribution_loc)
                intermediate_distribution_covs.append(intermediate_distribution_cov)
                intermediate_distribution_chols.append(intermediate_distribution_chol)

            intermediate_distribution_locs = torch.stack(
                intermediate_distribution_locs
            )  # [B * num_paths, 2]
            intermediate_distribution_covs = torch.stack(
                intermediate_distribution_covs
            )  # [B * num_paths, 2, 2]
            intermediate_distribution_chols = torch.stack(
                intermediate_distribution_chols
            )  # [B * num_paths, 2, 2]

            intermediate_distribution = D.MultivariateNormal(
                loc=intermediate_distribution_locs.unsqueeze(1),
                scale_tril=intermediate_distribution_chols.unsqueeze(1),
            )  # batch_shape: [B * num_paths, 1]

            intermediate_candidates_probs = intermediate_distribution.log_prob(
                waypoint_samples_candidates
            ).softmax(
                -1
            )  # [B * num_paths, 10000]
            candidates_probs = (
                waypoint_samples_candidates_probs * intermediate_candidates_probs
            )  # multiply probs  [B * num_paths, 10000]

            if g_num // num_goals == 0:
                # For first traj samples use the most probable candidate
                best_candidate_ind = candidates_probs.max(-1)[1]  # [B * num_paths]
                waypoint_samples = torch.gather(
                    waypoint_samples_candidates,
                    1,
                    best_candidate_ind.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2),
                ).squeeze(
                    1
                )  # [B * num_paths, 2]
            else:
                # For the other traj samples sample based on probability
                candidates_distr = D.Categorical(logits=candidates_probs)
                best_candidate_ind = candidates_distr.sample(sample_shape=(1,)).squeeze(
                    0
                )
                waypoint_samples = torch.gather(
                    waypoint_samples_candidates,
                    1,
                    best_candidate_ind.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2),
                ).squeeze(
                    1
                )  # [B * num_paths, 2]
            """
            take prev_waypoint_samples from here
            prev_waypoint_samples = waypoint_samples
            !!!!Potentially without these permutations and squeezing - check
            """
            waypoint_list.append(waypoint_samples)

        waypoint_list = waypoint_list[::-1]  # reverse the order
        waypoint_list = torch.stack(waypoint_list).permute(
            1, 0, 2
        )  # permute back to [B * num_paths, num_waypoints, 2]
        waypoint_samples_list.append(waypoint_list)

    waypoint_samples = torch.stack(
        waypoint_samples_list
    )  # [num_goals * num_traj, B * num_paths, num_waypoints, 2]
    # CWS End
    return waypoint_samples
