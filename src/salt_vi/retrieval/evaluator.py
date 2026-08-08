import numpy as np
import torch

from salt_vi.utils import eval_sysu


def _extract(loader, encoder, model, device):
    features = []
    with torch.no_grad():
        for batch_dict in loader:
            features.append(encoder(model, batch_dict, device).detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def evaluate_sysu(model, loader, device, backend):
    model.set_eval()
    query_features = _extract(loader.query_loader, backend.encode_query, model, device)

    trial_metrics = []
    for trial, gallery_loader in enumerate(loader.gallery_loaders):
        gallery_features = _extract(gallery_loader, backend.encode_gallery, model, device)
        similarity = query_features @ gallery_features.T
        trial_metrics.append(
            eval_sysu(
                -similarity,
                loader.query_label,
                loader.gallery_labels[trial],
                loader.query_cam,
                loader.gallery_cams[trial],
            )
        )

    cmc = np.mean([metrics[0] for metrics in trial_metrics], axis=0)
    m_ap = float(np.mean([metrics[1] for metrics in trial_metrics]))
    m_inp = float(np.mean([metrics[2] for metrics in trial_metrics]))
    return {backend.RESULT_KEY: (m_inp, m_ap, cmc)}
