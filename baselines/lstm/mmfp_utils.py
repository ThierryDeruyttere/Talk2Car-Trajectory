# repurposed metrics code from Multimodal Future Prediction
import numpy as np
from wemd import computeWEMD
from sklearn.mixture import GaussianMixture as GMM

# returns the closest hypothesis to the ground truth (oracle selection)
def get_best_hyp(hyps, gt):
    num_hyps = len(hyps)
    gts = np.stack([gt for i in range(0, num_hyps)], axis=1)  # n,num,c,1,1
    hyps = np.stack(hyps, axis=1)  # n,num,c,1,1

    def spatial_error(hyps, gts):
        diff = np.square(hyps - gts)  # n,num,c,1,1
        channels_sum = np.sum(diff, axis=2)  # n,num,1,1
        spatial_epes = np.sqrt(channels_sum)  # n,num,1,1
        return np.expand_dims(spatial_epes, axis=2)  # n,num,1,1,1

    def get_best(hypotheses, errors, num_hyps):
        indices = np.argmin(errors, axis=1)  # n,1,1,1
        shape = indices.shape
        # compute one-hot encoding
        encoding = np.zeros((shape[0], num_hyps, shape[1], shape[2], shape[3]))
        encoding[
            np.arange(shape[0]),
            indices,
            np.arange(shape[1]),
            np.arange(shape[2]),
            np.arange(shape[3]),
        ] = 1  # n,num,1,1,1

        hyps_channels = hypotheses.shape[2]
        encoding = np.concatenate(
            [encoding for i in range(hyps_channels)], axis=2
        )  # n,num,c,1,1
        reduced = hypotheses * encoding  # n,num,c,1,1
        reduced = np.sum(reduced, axis=1)  # n,c,1,1
        return reduced

    errors = spatial_error(hyps, gts)  # n,num,1,1,1
    best = get_best(hyps, errors, num_hyps)  # n,c,1,1
    return best


# compute the final displacement error between a hypothesis and ground truth
def get_FDE(hyp, gt):
    diff = np.square(hyp[:, 0:2, :, :] - gt[:, 0:2, :, :])
    channels_sum = np.sum(diff, axis=1)
    spatial_epe = np.sqrt(channels_sum)
    fde = np.mean(spatial_epe)
    return fde


# compute the final displacement error between the best hypothesis and the ground truth
def compute_oracle_FDE(hyps, gt):
    #     print("fde", hyps.shape, gt.shape)
    gt_loc = np.transpose(gt[:, :, 0:2, :], [0, 2, 1, 3])
    best_hyp = get_best_hyp(hyps, gt_loc)
    return get_FDE(best_hyp, gt_loc)


def wemd_from_samples(samples_1, samples_2, bins=512):
    hist_1, *_ = np.histogram2d(
        samples_1[:, 0], samples_1[:, 1], bins=np.linspace(0, bins, bins)
    )
    hist_2, *_ = np.histogram2d(
        samples_2[:, 0], samples_2[:, 1], bins=np.linspace(0, bins, bins)
    )
    return computeWEMD(hist_1, hist_2)


def wemd_from_pred_samples(y_pred,):
    gmm = GMM(covariance_type="diag")
    gmm = gmm.fit(y_pred)
    y_s, _ = gmm.sample(len(y_pred))
    return wemd_from_samples(y_s, y_pred)
